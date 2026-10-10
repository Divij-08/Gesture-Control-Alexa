"""Live runner: wake gate + fingerspelling MLP -> Session -> composed utterance.

The full front-to-back demo of the flow:
    open palm (WAKE) -> [command gloss] -> thumb_up (SPELL) -> fingerspell letters
    -> thumb_up (leave SPELL, flush word) -> thumb_left (STOP) -> compose -> print

Two MediaPipe pipelines run per frame, on DIFFERENT frame orientations (this is
load-bearing — see capture/wake_gate.py):
  * Holistic on the FLIPPED (mirror) frame  -> the control gate (WAKE/SPELL/STOP).
    The gate's thumb_left="left" rule assumes the selfie view.
  * Hands on the RAW (unflipped) frame       -> the recognizers, matching
    capture/landmarks.py's train==test invariant. While LISTENING:
      - normal mode  -> both-hands WORD model (recognizer/words) -> lexical glosses
      - spell mode   -> single-hand LETTER model (recognizer/fingerspell) -> letters
    Both feed the one Session token buffer in signing order.

MODES
  --sim         Scripted control sequence (no camera, no model). Proves gate-event
                -> Session -> compose end to end. Uses --command and --letters.
  (default)     Live webcam. Recognizes command words live (no keypress stand-in).

USAGE
  # simulated (works anywhere; add GESTUREACA_COMPOSER_OFFLINE=1 to skip Bedrock):
  python -m orchestrator.wake_session --sim --command NAV --letters CUBBONPARK

  # live webcam (needs recognizer/fingerspell + recognizer/words models + camera):
  #   open palm 1s = wake | sign a command word | thumb-up = toggle SPELL |
  #   fingerspell a proper noun | thumb-left 1s = stop -> compose
  python -m orchestrator.wake_session
"""
from __future__ import annotations

import argparse
import os
import time

from orchestrator.session import Session


# ── simulated run (no camera) ────────────────────────────────────────

def run_sim(command: str | None, letters: str, wake_word: str,
            enabled_speak: bool = True) -> str:
    """Drive Session with a scripted gate-event sequence and print each step."""
    sess = Session(wake_word=wake_word)

    print("\n[sim] open palm -> WAKE")
    sess.on_wake()

    if command:
        print(f"[sim] command gloss -> {command!r}")
        sess.add_command(command)

    print("[sim] thumb_up -> enter SPELL")
    sess.on_spell_toggle(True)
    for ch in letters:
        sess.on_letter(ch)
    print(f"[sim] fingerspelled letters -> {letters!r}")

    print("[sim] thumb_up -> leave SPELL (flush word)")
    sess.on_spell_toggle(False)

    print(f"[sim] TokenBuffer -> {sess.tokens + ([sess._current_word] if sess._current_word else [])}")
    print("[sim] thumb_left -> STOP -> compose")
    utterance = sess.on_stop()
    print(f"\n  >>> UTTERANCE: {utterance}\n")

    # Speak the final composer output. Synchronous here so audio finishes before
    # the process exits (no background thread as in the live loop).
    if enabled_speak and utterance:
        try:
            from speak import speak
            speak(utterance)
        except Exception as e:  # noqa: BLE001 — TTS must not break the run
            print(f"[voice] speak failed ({type(e).__name__}: {e}); continuing.")
    return utterance


# ── live webcam run ──────────────────────────────────────────────────

def run_live(wake_word: str, enabled_speak: bool = True,
             fullscreen: bool = False, screen=(1920, 1080)) -> None:
    import cv2
    import numpy as np
    import joblib
    import mediapipe as mp

    from capture.wake_gate import ListeningGate
    from capture.landmarks import (make_hands, extract_landmarks,
                                   extract_landmarks_both_hands)
    from orchestrator.shortcuts import SHORTCUTS, utterance_for as shortcut_utterance
    from ui.overlay import draw_hud, letterbox_to

    # Fingerspelling model (letters, single-hand 63-d) — used in SPELL mode.
    letter_path = os.path.join("recognizer", "fingerspell", "model.joblib")
    if not os.path.exists(letter_path):
        raise SystemExit(f"No model at {letter_path}. Run: python -m training.train_fingerspell")
    lb = joblib.load(letter_path)
    letter_clf, letter_labels = lb["model"], lb["labels"]

    # Word model (lexical glosses, both-hands 126-d) — used in normal LISTENING mode.
    # Replaces the old --command keypress stand-in with live recognition. Optional:
    # if the word model is absent we fall back to the keypress stand-in.
    word_path = os.path.join("recognizer", "words", "model.joblib")
    word_clf = word_labels = None
    if os.path.exists(word_path):
        wb = joblib.load(word_path)
        word_clf, word_labels = wb["model"], wb["labels"]
    else:
        print(f"[warn] no word model at {word_path}; command glosses via 'c' keypress only.")

    import threading
    from speak import speak

    LETTER_CONF, LETTER_STABLE = 0.80, 15
    WORD_CONF, WORD_STABLE = 0.75, 12
    sess = Session(wake_word=wake_word)

    def speak_out(utterance):
        """Speak the final utterance aloud on a background thread (never blocks the
        camera loop) and never crashes it (TTS/Polly/creds failures are swallowed)."""
        if not enabled_speak or not utterance:
            return
        def _run():
            try:
                speak(utterance)
            except Exception as e:  # noqa: BLE001 — TTS must not kill the session
                print(f"[voice] speak failed ({type(e).__name__}: {e}); continuing.")
        threading.Thread(target=_run, daemon=True).start()

    # Last utterance for the HUD banner (persists after STOP / shortcut so it stays
    # readable during the demo). Mutable container so callbacks can update it.
    hud = {"utterance": None}

    def on_stop():
        utterance = sess.on_stop()          # composer output ("" if buffer was empty)
        if utterance:
            print(f"[STOP] -> {utterance}")
            hud["utterance"] = utterance
            speak_out(utterance)            # <-- composer output goes to speak()
        else:
            print("[STOP] (empty buffer — nothing to compose or speak)")

    def fire_shortcut(gesture):
        # Fires in LISTENING (after WAKE). Ends the session: discard any tokens the
        # driver had started, dispatch the shortcut's fixed utterance. The gate has
        # already flipped to IDLE; mirror that on the Session so state stays in sync.
        utterance = shortcut_utterance(gesture, wake_word)
        print(f"[SHORTCUT] {gesture} -> {utterance}  (session ended)")
        hud["utterance"] = utterance
        sess.tokens = []
        sess._current_word = ""
        sess.active = False
        speak_out(utterance)                # <-- shortcut output goes to speak()

    def on_wake():
        sess.on_wake()
        hud["utterance"] = None             # clear the banner for a fresh session
        print("[WAKE] listening")

    gate = ListeningGate(
        on_wake=on_wake,
        on_spell_toggle=lambda m: (sess.on_spell_toggle(m), print(f"[SPELL] spell_mode -> {m}")),
        on_stop=on_stop,
        on_shortcut=fire_shortcut,
        shortcut_gestures=SHORTCUTS.keys(),
        # frame hand-off handled inline below so we can route to letter vs word MLP
    )

    win = "GestureACA  (q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")
    # Word model needs BOTH hands (126-d). Fingerspelling needs exactly ONE hand:
    # extract_landmarks takes results[0], so with a 2-hand detector the "first" hand
    # could be the wrong one (other hand / ghost) — that caused left-hand misses and
    # O being confused with the other thumb. Use a separate 1-hand detector for spell.
    hands = make_hands(max_num_hands=2)      # word recognition (LISTENING, normal)
    hands1 = make_hands(max_num_hands=1)     # letter recognition (SPELL mode)

    # Independent debounce state for letters and words.
    l_last, l_stable, l_recent = None, 0, None
    w_last, w_stable, w_recent = None, 0, None
    print("open palm 1s = WAKE | thumb-up = SPELL toggle | thumb-left 1s = STOP | q = quit")
    print("after WAKE, shortcuts (hold 1s, ends session): " +
          " | ".join(f"{g}->{u}" for g, u in
                     ((g, shortcut_utterance(g, wake_word)) for g in SHORTCUTS)))
    mp_drawing = mp.solutions.drawing_utils
    try:
        # Holistic's built-in hand detector is weaker than the dedicated Hands model;
        # bump model_complexity and lower thresholds so it finds hands more reliably.
        with mp_holistic.Holistic(model_complexity=1,
                                  min_detection_confidence=0.3,
                                  min_tracking_confidence=0.3) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                # Holistic on the FLIPPED frame -> the control gate.
                flipped = cv2.flip(frame, 1)
                holo = holistic.process(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
                gate.update(holo, time.time())

                # Always draw what the GATE sees (Holistic hands), in every state,
                # so IDLE gives feedback on whether your hand is detected.
                holo_hand_seen = bool(holo.right_hand_landmarks or holo.left_hand_landmarks)
                for hlm in (holo.right_hand_landmarks, holo.left_hand_landmarks):
                    if hlm:
                        mp_drawing.draw_landmarks(flipped, hlm, mp_holistic.HAND_CONNECTIONS)

                # Recognition on the RAW (unflipped) frame -> Hands MLPs (train==test).
                if sess.active:
                    rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if sess.spell_mode:
                        # SPELL mode: 1-hand detector -> letter MLP -> committed letters.
                        feats = extract_landmarks(hands1.process(rgb_raw))
                        if feats is not None:
                            proba = letter_clf.predict_proba([feats])[0]
                            pred, conf = letter_labels[int(np.argmax(proba))], float(proba.max())
                            if conf >= LETTER_CONF:
                                if pred == l_last:
                                    l_stable += 1
                                else:
                                    l_last, l_stable, l_recent = pred, 1, None
                                if l_stable == LETTER_STABLE and pred != l_recent:
                                    sess.on_letter(pred)
                                    l_recent = pred
                            else:
                                l_last, l_stable = None, 0
                    elif word_clf is not None and gate.current_gesture not in SHORTCUTS:
                        # Normal LISTENING: 2-hand detector -> word MLP -> lexical gloss.
                        # Skip while a SHORTCUT pose is being held so it can't commit a
                        # spurious word before the shortcut fires.
                        feats = extract_landmarks_both_hands(hands.process(rgb_raw))
                        if feats is not None:
                            proba = word_clf.predict_proba([feats])[0]
                            pred, conf = word_labels[int(np.argmax(proba))], float(proba.max())
                            if conf >= WORD_CONF:
                                if pred == w_last:
                                    w_stable += 1
                                else:
                                    w_last, w_stable, w_recent = pred, 1, None
                                if w_stable == WORD_STABLE and pred != w_recent:
                                    sess.add_command(pred)   # append lexical gloss token
                                    w_recent = pred
                                    print(f"[word] {pred!r} -> tokens {sess.tokens}")
                            else:
                                w_last, w_stable = None, 0
                    # NOTE: do NOT draw hres here — those landmarks are from the RAW
                    # (unflipped) frame, so drawing them on the flipped preview
                    # produced a mirror-image "ghost" skeleton. The Holistic hands
                    # (drawn above on the flipped frame) already show the real hand.

                # ── demo HUD: build the display state, hand off to ui.overlay ──
                if not sess.active:
                    mode = "IDLE"
                elif sess.spell_mode:
                    mode = "SPELLING"
                else:
                    mode = "LISTENING"
                g = gate.current_gesture
                need = gate.spell_hold_sec if g == "thumb_up" else gate.hold_sec
                hold = (gate.hold_seconds / need) if (g and need) else 0.0
                state = {
                    "mode": mode,
                    "tokens": list(sess.tokens),
                    "spelling": sess._current_word,
                    "utterance": hud["utterance"],
                    "gesture": g,
                    "hold": hold,
                    "hand_seen": holo_hand_seen,
                }
                canvas = draw_hud(flipped, state)   # returns camera+panel canvas
                if fullscreen:
                    # pad to the screen aspect with white so fullscreen shows no
                    # black letterbox bars (they blend into the white panel instead)
                    canvas = letterbox_to(canvas, screen[0], screen[1])
                cv2.imshow(win, canvas)

                if (cv2.waitKey(5) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        hands.close()
        hands1.close()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="scripted run, no camera")
    ap.add_argument("--command", default="NAV",
                    help="[--sim] stand-in command gloss (live mode recognizes words instead)")
    ap.add_argument("--letters", default="CUBBONPARK", help="[--sim] fingerspelled word")
    ap.add_argument("--wake-word", default=os.environ.get("GESTUREACA_WAKE_WORD", "Hey BMW"))
    ap.add_argument("--no-speak", action="store_true",
                    help="don't speak the utterance aloud (Polly TTS is on by default)")
    ap.add_argument("--fullscreen", action="store_true",
                    help="fullscreen window, white-padded to the screen aspect (no black bars)")
    ap.add_argument("--screen", default="1920x1080",
                    help="[--fullscreen] screen WxH for aspect padding (default 1920x1080)")
    args = ap.parse_args()

    if args.sim:
        run_sim(args.command or None, args.letters, args.wake_word,
                enabled_speak=not args.no_speak)
    else:
        try:
            sw, sh = (int(v) for v in args.screen.lower().split("x"))
        except ValueError:
            sw, sh = 1920, 1080
        run_live(args.wake_word, enabled_speak=not args.no_speak,
                 fullscreen=args.fullscreen, screen=(sw, sh))


if __name__ == "__main__":
    main()
