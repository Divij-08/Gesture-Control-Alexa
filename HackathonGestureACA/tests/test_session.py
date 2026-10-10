"""Tests for the wake Session logic and the ListeningGate state machine.

Fully offline: Session uses a stub composer (no LLM), and the gate is driven with
fake MediaPipe-shaped results (no camera). Run:
    python -m pytest tests/test_session.py -v
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.session import Session
from capture import wake_gate


# ── Session: buffer assembly + compose on stop ───────────────────────

def _echo_composer(tokens, wake_word):
    """Deterministic stub: mimic the shape without an LLM."""
    return f"{wake_word}::{'|'.join(tokens)}"


def test_command_then_spelled_word_interleave_in_order():
    s = Session(wake_word="Hey BMW", composer=_echo_composer)
    s.on_wake()
    s.add_command("NAV")
    s.on_spell_toggle(True)
    for ch in "CUBBONPARK":
        s.on_letter(ch)
    s.on_spell_toggle(False)          # flush -> one token
    out = s.on_stop()
    assert s.tokens == ["NAV", "CUBBONPARK"]
    assert out == "Hey BMW::NAV|CUBBONPARK"


def test_stop_flushes_unterminated_spell_word():
    # User stops while still in spell mode — pending word must still be flushed.
    s = Session(composer=_echo_composer)
    s.on_wake()
    s.on_spell_toggle(True)
    for ch in "TAJMAHAL":
        s.on_letter(ch)
    out = s.on_stop()                 # no explicit leave-spell; stop flushes
    assert s.tokens == ["TAJMAHAL"]
    assert out.endswith("TAJMAHAL")


def test_letters_ignored_outside_spell_mode():
    s = Session(composer=_echo_composer)
    s.on_wake()
    s.on_letter("A")                  # not in spell mode -> ignored
    s.on_spell_toggle(True)
    s.on_letter("B")
    s.on_spell_toggle(False)
    s.on_stop()
    assert s.tokens == ["B"]


def test_command_ignored_during_spell_mode():
    s = Session(composer=_echo_composer)
    s.on_wake()
    s.on_spell_toggle(True)
    s.add_command("NAV")              # commands don't interrupt a spelled word
    s.on_letter("X")
    s.on_spell_toggle(False)
    s.on_stop()
    assert s.tokens == ["X"]


def test_wake_resets_previous_session():
    s = Session(composer=_echo_composer)
    s.on_wake()
    s.add_command("CALL")
    s.on_stop()
    s.on_wake()                       # fresh session
    assert s.tokens == []
    assert s.active is True


def test_empty_session_skips_composer():
    # Woke then stopped with no signs: composer must NOT be called (empty gloss
    # list makes the LLM hallucinate). on_stop returns "" so callers skip dispatch.
    calls = []
    def spy_composer(tokens, ww):
        calls.append(tokens)
        return "SHOULD NOT HAPPEN"
    s = Session(composer=spy_composer)
    s.on_wake()
    out = s.on_stop()
    assert s.tokens == []
    assert out == ""                  # nothing composed
    assert calls == []                # composer never invoked on empty buffer


def test_compose_empty_never_calls_llm(monkeypatch):
    # Defense-in-depth: compose([]) must not hit Bedrock even if called directly.
    from composer import bedrock
    monkeypatch.delenv("GESTUREACA_COMPOSER_OFFLINE", raising=False)
    called = []
    monkeypatch.setattr(bedrock, "_make_client",
                        lambda *a, **k: called.append(True) or (_ for _ in ()).throw(AssertionError))
    from composer.compose import compose
    assert compose([]) == "Hey BMW."          # rule-based wake-word-only
    assert compose(["", "   "]) == "Hey BMW." # blank tokens count as empty
    assert called == []                        # Bedrock client never built


# ── ListeningGate: fake-result driver, no camera ─────────────────────

def _fake_results(gesture: str | None, wrist=(0.5, 0.5)):
    """Build a MediaPipe-Holistic-shaped result whose right hand yields `gesture`.

    We synthesize 21 hand landmarks positioned so wake_gate's geometric rules
    classify them as the desired control gesture. Coordinates are in normalized
    image space (selfie/mirror view, as the gate expects).
    """
    def L(x, y):
        return SimpleNamespace(x=float(x), y=float(y), z=0.0, visibility=1.0)

    wx, wy = wrist
    lm = [L(wx, wy)] * 21          # default all at wrist (curled/tucked)
    lm[0] = L(wx, wy)              # wrist
    lm[9] = L(wx, wy - 0.20)       # middle MCP -> gives a hand_scale of 0.20
    lm[5] = L(wx, wy - 0.15)       # index MCP (thumb-extended reference)

    def extend(tip, pip, dx, dy):
        lm[pip] = L(wx + dx * 0.10, wy + dy * 0.10)
        lm[tip] = L(wx + dx * 0.22, wy + dy * 0.22)   # tip farther from wrist than pip

    if gesture == "open":
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            extend(tip, pip, 0, -1)                    # 4 fingers up
        lm[4] = L(wx + 0.20, wy - 0.05)                # thumb clearly out
    elif gesture == "thumb_up":
        lm[4] = L(wx, wy - 0.35)                        # thumb tip far ABOVE wrist (and far from index MCP)
    elif gesture == "thumb_left":
        lm[4] = L(wx - 0.22, wy + 0.02)                # thumb tip well LEFT of wrist
    elif gesture == "shaka":
        extend(20, 18, 0, -1)                          # pinky up
        lm[4] = L(wx + 0.20, wy - 0.05)                # thumb clearly out (index/middle/ring stay curled)
    elif gesture == "peace":
        extend(8, 6, 0, -1)                            # index up
        extend(12, 10, 0, -1)                          # middle up  (ring/pinky curled, thumb tucked)
    elif gesture == "point":
        extend(8, 6, 0, -1)                            # index only up
    # gesture is None -> plain tucked fist (thumb at wrist), classifies as None

    hand = SimpleNamespace(landmark=lm)
    return SimpleNamespace(right_hand_landmarks=hand, left_hand_landmarks=None,
                           pose_landmarks=None)


def test_gesture_classifier_labels():
    assert wake_gate.hand_gesture(_fake_results("open")) == "open"
    assert wake_gate.hand_gesture(_fake_results("thumb_up")) == "thumb_up"
    assert wake_gate.hand_gesture(_fake_results("thumb_left")) == "thumb_left"
    assert wake_gate.hand_gesture(_fake_results(None)) is None


def test_shortcut_gesture_classifier_labels():
    assert wake_gate.hand_gesture(_fake_results("shaka")) == "shaka"
    assert wake_gate.hand_gesture(_fake_results("peace")) == "peace"
    assert wake_gate.hand_gesture(_fake_results("point")) == "point"


def test_shortcuts_do_not_collide_with_wake():
    # A shortcut pose must NOT be classified as the open-palm WAKE.
    for g in ("shaka", "peace", "point"):
        assert wake_gate.hand_gesture(_fake_results(g)) != "open"


def test_shortcut_does_not_fire_from_idle():
    # Shortcuts now require WAKE first — holding one in IDLE must do nothing.
    fired = []
    gate = wake_gate.ListeningGate(
        hold_sec=1.0, cooldown_sec=1.0,
        shortcut_gestures={"shaka", "peace", "point"},
        on_shortcut=lambda g: fired.append(g),
    )
    t = 50.0
    for _ in range(20):
        gate.update(_fake_results("shaka"), t); t += 0.1
    assert fired == []
    assert gate.state == "IDLE"


def test_shortcut_fires_in_listening_and_ends_session():
    fired, events = [], []
    gate = wake_gate.ListeningGate(
        hold_sec=1.0, min_listen_sec=1.0, cooldown_sec=1.0,
        shortcut_gestures={"shaka", "peace", "point"},
        on_wake=lambda: events.append("wake"),
        on_shortcut=lambda g: fired.append(g),
        on_stop=lambda: events.append("stop"),
    )
    t = 100.0
    # WAKE first.
    for _ in range(20):
        gate.update(_fake_results("open"), t); t += 0.1
    assert gate.state == "LISTENING" and "wake" in events

    # Past cooldown + grace, hold "peace" > 1s in LISTENING -> fires, ends session.
    t += 1.2
    for _ in range(15):
        gate.update(_fake_results("peace"), t); t += 0.1
    assert fired == ["peace"]
    assert gate.state == "IDLE"           # session ended
    assert "stop" not in events           # shortcut path, not the STOP callback


def test_shortcut_ignored_in_spell_mode():
    # A shortcut pose while spelling must NOT fire (letters take priority there).
    fired = []
    gate = wake_gate.ListeningGate(
        hold_sec=1.0, spell_hold_sec=0.8, min_listen_sec=1.0, cooldown_sec=1.0,
        shortcut_gestures={"shaka", "peace", "point"},
        on_shortcut=lambda g: fired.append(g),
    )
    t = 200.0
    for _ in range(20):                       # WAKE
        gate.update(_fake_results("open"), t); t += 0.1
    t += 1.2
    for _ in range(12):                       # thumb_up -> spell_mode ON
        gate.update(_fake_results("thumb_up"), t); t += 0.1
    assert gate.spell_mode is True
    t += 1.2
    for _ in range(15):                       # hold "peace" while spelling
        gate.update(_fake_results("peace"), t); t += 0.1
    assert fired == []                        # suppressed in spell mode
    assert gate.state == "LISTENING"


def test_shortcut_registry_utterances():
    from orchestrator import shortcuts
    assert shortcuts.is_shortcut("shaka")
    assert not shortcuts.is_shortcut("open")
    assert shortcuts.utterance_for("shaka", "Hey BMW") == "Hey BMW, play some music"
    assert shortcuts.utterance_for("peace", "Hey BMW").startswith("Hey BMW, ")
    # body carries no wake word; utterance_for prepends it
    assert "Hey BMW" not in shortcuts.SHORTCUTS["point"]


def test_gate_wake_spell_stop_sequence():
    events = []
    gate = wake_gate.ListeningGate(
        hold_sec=1.0, spell_hold_sec=0.8, min_listen_sec=1.0, cooldown_sec=1.0,
        on_wake=lambda: events.append("wake"),
        on_spell_toggle=lambda m: events.append(f"spell:{m}"),
        on_stop=lambda: events.append("stop"),
        on_frame=lambda vec, spell: events.append(f"frame:{spell}"),
    )
    t = 100.0

    # Hold open palm > 1s still -> WAKE
    for _ in range(20):
        gate.update(_fake_results("open"), t); t += 0.1
    assert "wake" in events and gate.state == "LISTENING"

    # A neutral signing frame gets handed off (spell_mode False)
    events.clear()
    gate.update(_fake_results(None), t); t += 0.1
    assert events == ["frame:False"]

    # Advance past cooldown + grace, then hold thumb_up > 0.8s -> toggle spell True
    t += 1.2
    events.clear()
    for _ in range(12):
        gate.update(_fake_results("thumb_up"), t); t += 0.1
    assert "spell:True" in events and gate.spell_mode is True

    # Past cooldown, hold thumb_left > 1s -> STOP
    t += 1.2
    events.clear()
    for _ in range(15):
        gate.update(_fake_results("thumb_left"), t); t += 0.1
    assert "stop" in events and gate.state == "IDLE"


def test_moving_hand_never_triggers():
    """A gesture that drifts (never still) must not complete the hold."""
    gate = wake_gate.ListeningGate(hold_sec=1.0)
    fired = []
    gate.on_wake = lambda: fired.append("wake")
    t = 0.0
    for i in range(30):
        # wrist drifts each frame beyond still_eps -> streak keeps resetting
        gate.update(_fake_results("open", wrist=(0.3 + i * 0.03, 0.5)), t)
        t += 0.1
    assert fired == []
    assert gate.state == "IDLE"
