"""Quick shortcuts — one gesture -> one fixed ACA utterance, no session, no LLM.

A shortcut is the most reliable path in the system: a single IDLE gesture maps
directly to a complete, fixed utterance, bypassing BOTH fingerspelling and the
LLM composer. No fingerspelling to get wrong, no proper noun, no LLM latency.

The gesture DETECTION is training-free geometric rules in capture/wake_gate.py
(shaka / peace / point). This module owns only the gesture-id -> utterance-body
mapping. Bodies are stored WITHOUT the wake word; utterance_for() prepends the
configured wake word so the result is ACA-triggerable — ACA only wakes on its
exact wake word (design §8/§16), same rule the composer follows.

Shortcuts fire in LISTENING (after WAKE) — the "Alexa is ready, now gesture"
model — and each ends the session after dispatching. Re-map freely: it's just
this dict. (Detection is suppressed during spell mode; the word recognizer is
paused while a shortcut pose is held so it can't commit a spurious token.)
"""
from __future__ import annotations

import os

# gesture-id (from capture.wake_gate.hand_gesture) -> fixed utterance BODY (no wake word)
SHORTCUTS: dict[str, str] = {
    "shaka": "play some music",
    "peace": "what's the weather today?",
    "point": "switch on the AC",
}


def default_wake_word() -> str:
    """The configured wake word (matches the composer's config)."""
    return os.environ.get("GESTUREACA_WAKE_WORD", "Hey BMW")


def is_shortcut(gesture: str | None) -> bool:
    """True if this gesture-id is bound to a shortcut."""
    return gesture in SHORTCUTS


def utterance_for(gesture: str, wake_word: str | None = None) -> str:
    """Return the full wake-word-prefixed utterance for a shortcut gesture.

    >>> utterance_for("shaka", wake_word="Hey BMW")
    'Hey BMW, play some music'

    Raises KeyError if the gesture isn't a registered shortcut (callers should
    gate with is_shortcut first).
    """
    body = SHORTCUTS[gesture]
    ww = wake_word or default_wake_word()
    return f"{ww}, {body}"
