"""Training-free camera control gate — WAKE / SPELL-toggle / STOP.

Vendored from a standalone prototype (~/Downloads/wake_sign/wake_sign_gate.py) and
trimmed to the reusable core: the geometric gesture rules + the ListeningGate state
machine. The standalone OpenCV runner is intentionally dropped — our orchestrator
(orchestrator/wake_session.py) owns the camera loop and feeds this gate.

There is NO ML model for the three control gestures — they are hardcoded geometric
rules on MediaPipe landmarks (see the prototype's COMPARISON_EXPLAINED.md). Control
gestures must be 100% predictable and need no training. The rules are rotation- and
scale-robust, and a gesture must be held STILL for ~1s to fire, so normal signing
(which moves) can't false-trigger.

State machine (design §7 WAKE / SPELL / §16 STOP):
    IDLE      --open palm (held 1s still)-->  LISTENING, spell_mode=False   (WAKE)
    LISTENING --thumb_up (held 0.8s)------->  spell_mode = not spell_mode   (SPELL toggle)
    LISTENING --thumb_left (held 1s)------->  IDLE                          (STOP)

IMPORTANT — landmark pipelines differ:
  * This gate uses MediaPipe **Holistic** and works on the MIRRORED (flipped) frame
    — the `thumb_left` rule assumes the selfie-view (dx<0 = pointing left on screen).
  * Our fingerspelling MLP uses MediaPipe **Hands** on the UNFLIPPED frame
    (capture/landmarks.py, the train==test invariant). The runner therefore feeds
    Holistic the flipped frame and Hands the raw frame — do NOT feed this gate an
    unflipped frame or `thumb_left` inverts.

Works under our pinned mediapipe==0.10.21 (Holistic is present); the prototype's
0.10.14 pin is not required.
"""
from __future__ import annotations

import math

import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

POSE_N, HAND_N = 33, 21


# ── landmark vector (kept for parity with the CTC path's 258-d input) ─────────

def extract_keypoints(results) -> np.ndarray:
    """MediaPipe Holistic result -> flat (258,) float32 vector (zeros where absent).

    pose 33*4 + left hand 21*3 + right hand 21*3 = 132 + 63 + 63 = 258.
    This is the vector the CTC lexical path will consume; the gate itself only
    needs the hand geometry below, but we expose it so `send()` can hand it off.
    """
    pose = (np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(POSE_N * 4))
    lh = (np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(HAND_N * 3))
    rh = (np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(HAND_N * 3))
    return np.concatenate([pose, lh, rh]).astype(np.float32)


# ── rotation/scale-robust hand geometry (the hardcoded control rules) ─────────

def _d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _hand_scale(lm):
    """Stable hand-size reference (wrist -> middle MCP) for scale-invariant thresholds."""
    return max(_d(lm[0], lm[9]), 1e-6)


def _finger_extended(lm, tip, pip):
    """Extended = tip farther from wrist than pip (rotation-invariant, 5% margin)."""
    return _d(lm[tip], lm[0]) > _d(lm[pip], lm[0]) * 1.05


def _four_fingers(lm):
    return [_finger_extended(lm, 8, 6), _finger_extended(lm, 12, 10),
            _finger_extended(lm, 16, 14), _finger_extended(lm, 20, 18)]


def _thumb_extended(lm):
    return _d(lm[4], lm[5]) > 0.55 * _hand_scale(lm)


def _thumb_dir(lm):
    return lm[4].x - lm[0].x, lm[4].y - lm[0].y


def hand_gesture(results):
    """Classify the dominant hand into a reserved control gesture, or None.

    CONTROL gestures (fire in LISTENING — design §7/§16):
        'open'       = 4 fingers + thumb extended (open palm)  -> WAKE
        'thumb_up'   = fingers curled, thumb UP                -> SPELL toggle
        'thumb_left' = fingers curled, thumb LEFT (mirrored)   -> STOP

    SHORTCUT gestures (fire in LISTENING, after WAKE — the "Alexa is ready, now
    gesture" model; each dispatches a fixed utterance and ends the session; see
    orchestrator/shortcuts.py):
        'shaka'      = thumb + pinky out, middle 3 curled      -> shortcut
        'peace'      = index + middle up, ring + pinky curled   -> shortcut
        'point'      = index only up                            -> shortcut

        None         = anything else (normal signing) -> ignored
    """
    hand = results.right_hand_landmarks or results.left_hand_landmarks
    if hand is None:
        return None
    lm = hand.landmark
    fingers = _four_fingers(lm)          # [index, middle, ring, pinky]
    idx_up, mid_up, ring_up, pinky_up = fingers
    n = sum(fingers)
    thumb_out = _thumb_extended(lm)

    # ── control gestures ──────────────────────────────────────────
    if n == 4 and thumb_out:
        return "open"

    if n == 0 and thumb_out:
        dx, dy = _thumb_dir(lm)
        if math.hypot(dx, dy) < 0.10:
            return None
        if dy < 0 and abs(dy) > 1.3 * abs(dx):
            return "thumb_up"
        if dx < 0 and abs(dx) > 1.3 * abs(dy):
            return "thumb_left"
        return None

    # ── shortcut gestures (finger-count patterns; thumb direction not needed) ──
    # shaka: only thumb + pinky out (index/middle/ring curled)
    if thumb_out and pinky_up and not idx_up and not mid_up and not ring_up:
        return "shaka"

    # peace: index + middle up, ring + pinky curled (thumb may be in or out)
    if idx_up and mid_up and not ring_up and not pinky_up:
        return "peace"

    # point: index only up (middle/ring/pinky curled)
    if idx_up and not mid_up and not ring_up and not pinky_up:
        return "point"

    return None


def wrist_xy(results):
    """(x, y) of the dominant hand's wrist, or None (for the stillness check)."""
    hand = results.right_hand_landmarks or results.left_hand_landmarks
    if hand is None:
        return None
    w = hand.landmark[0]
    return (w.x, w.y)


# ── the gate state machine ────────────────────────────────────────────────────

class ListeningGate:
    """WAKE / SPELL-toggle / STOP state machine driven by held-still control gestures.

    Emits high-level events via callbacks so the orchestrator stays UI/loop-agnostic:
      on_wake()            -> a session started (spell_mode reset to False)
      on_spell_toggle(bool)-> spell_mode flipped; arg = the NEW spell_mode
      on_stop()            -> session ended
      on_frame(vec, spell) -> a normal LISTENING frame (258-d Holistic vector + flag)
      on_shortcut(gesture) -> a shortcut gesture was held in IDLE (arg = gesture-id)

    Shortcuts fire from LISTENING (after WAKE), NOT in spell mode, using the same
    hold-still + grace + cooldown machinery as STOP; firing ends the session (-> IDLE).
    `shortcut_gestures` is the set of gesture-ids to treat as hold-to-fire shortcuts
    (e.g. {"shaka","peace","point"}); the gate stays decoupled from the registry — it
    just reports which gesture fired, the runner maps it to an utterance.

    Robustness (see the prototype's COMPARISON_EXPLAINED.md §5): a control gesture must
    be held STILL for a wall-clock duration before firing; a cooldown prevents one hold
    from double-firing; a grace period stops STOP firing on the palm->sign transition.
    """

    def __init__(self, hold_sec=1.0, spell_hold_sec=0.8, still_eps=0.06,
                 min_listen_sec=1.0, cooldown_sec=1.0,
                 on_wake=None, on_spell_toggle=None, on_stop=None, on_frame=None,
                 on_shortcut=None, shortcut_gestures=None):
        self.state = "IDLE"
        self.hold_sec = hold_sec
        self.spell_hold_sec = spell_hold_sec
        self.still_eps = still_eps
        self.min_listen_sec = min_listen_sec
        self.cooldown_sec = cooldown_sec

        self.on_wake = on_wake
        self.on_spell_toggle = on_spell_toggle
        self.on_stop = on_stop
        self.on_frame = on_frame
        self.on_shortcut = on_shortcut
        self.shortcut_gestures = set(shortcut_gestures or ())

        self._streak_gesture = None
        self._streak_start = 0.0
        self._streak_anchor = None
        self._last_trigger_t = -1e9
        self._listen_start = 0.0

        self.spell_mode = False
        self.frames_sent = 0

        self.current_gesture = None
        self.hold_seconds = 0.0

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _update_streak(self, gesture, wrist, now):
        cont = (gesture is not None
                and gesture == self._streak_gesture
                and wrist is not None and self._streak_anchor is not None
                and self._dist(wrist, self._streak_anchor) <= self.still_eps)
        if not cont:
            self._streak_gesture = gesture
            self._streak_start = now
            self._streak_anchor = wrist
        self.current_gesture = gesture
        self.hold_seconds = (now - self._streak_start) if gesture is not None else 0.0

    def _held(self, gesture, now, want, need_sec):
        return gesture == want and (now - self._streak_start) >= need_sec

    def update(self, results, now):
        """Call once per camera frame with a MediaPipe Holistic result + wall-clock now."""
        gesture = hand_gesture(results)
        wrist = wrist_xy(results)
        self._update_streak(gesture, wrist, now)
        cooling = (now - self._last_trigger_t) < self.cooldown_sec

        if self.state == "IDLE":
            if self._held(gesture, now, "open", self.hold_sec) and not cooling:
                self.state = "LISTENING"
                self._last_trigger_t = now
                self._listen_start = now
                self.spell_mode = False
                self.frames_sent = 0
                if self.on_wake:
                    self.on_wake()

        elif self.state == "LISTENING":
            past_grace = (now - self._listen_start) >= self.min_listen_sec

            if self._held(gesture, now, "thumb_up", self.spell_hold_sec) and not cooling:
                self._last_trigger_t = now
                self.spell_mode = not self.spell_mode
                if self.on_spell_toggle:
                    self.on_spell_toggle(self.spell_mode)
                return self.state

            if self._held(gesture, now, "thumb_left", self.hold_sec) and past_grace and not cooling:
                self._last_trigger_t = now
                self.state = "IDLE"
                self.spell_mode = False
                if self.on_stop:
                    self.on_stop()
                return self.state

            # Shortcuts fire AFTER wake (in LISTENING) — the "Alexa is ready, now
            # gesture" model. A registered shortcut gesture held still (and NOT in
            # spell mode) dispatches its fixed utterance and ENDS the session (-> IDLE),
            # discarding any accumulated tokens. Grace + cooldown gated like STOP.
            if (gesture in self.shortcut_gestures and not self.spell_mode
                    and (now - self._streak_start) >= self.hold_sec
                    and past_grace and not cooling):
                self._last_trigger_t = now
                self.state = "IDLE"
                self.spell_mode = False
                if self.on_shortcut:
                    self.on_shortcut(gesture)
                return self.state

            # normal LISTENING frame -> hand off tagged with the current flag
            self.frames_sent += 1
            if self.on_frame:
                self.on_frame(extract_keypoints(results), self.spell_mode)

        return self.state


__all__ = ["ListeningGate", "hand_gesture", "wrist_xy", "extract_keypoints"]
