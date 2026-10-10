"""Session logic — turns wake-gate events + recognized letters into a composed utterance.

Pure, camera-free, and LLM-agnostic so it can be unit-tested without a webcam or
AWS. The live runner (orchestrator/wake_session.py) wires this to the real camera,
the wake gate, and the fingerspelling MLP; tests and --sim drive it directly.

Flow (design §7):
    WAKE            -> new empty session, spell_mode=False
    SPELL toggle    -> flip spell_mode; entering spell starts a fresh spelled word,
                       leaving spell flushes the accumulated word as ONE token
    letter (spell)  -> append to the current spelled word
    command gloss   -> append as a lexical token (stand-in for the CTC path, which
                       isn't built yet — supplied via set_command / the runner)
    STOP            -> flush any pending word, compose(tokens) -> utterance

Tokens accumulate in signing order, interleaved (design §7): a command gloss and a
spelled word land in the one buffer in the order they occurred.
"""
from __future__ import annotations

from typing import Callable, Optional

from composer.compose import compose


class Session:
    """Accumulates a TokenBuffer across one wake->stop cycle and composes on stop.

    `composer` defaults to composer.compose.compose but is injectable so tests can
    pass a stub and avoid any LLM/Bedrock call.
    """

    def __init__(self, wake_word: str = "Hey BMW",
                 composer: Callable[[list[str], str], str] = compose):
        self.wake_word = wake_word
        self._composer = composer
        self.tokens: list[str] = []
        self.spell_mode = False
        self._current_word = ""
        self.active = False
        self.last_utterance: Optional[str] = None

    # ── gate events ──────────────────────────────────────────────

    def on_wake(self) -> None:
        """Start a fresh session (clears any previous buffer)."""
        self.tokens = []
        self._current_word = ""
        self.spell_mode = False
        self.active = True
        self.last_utterance = None

    def on_spell_toggle(self, new_spell_mode: bool) -> None:
        """Flip spell mode. Leaving spell mode flushes the spelled word as one token."""
        if not new_spell_mode:
            self._flush_word()          # spell -> normal: commit the word we just spelled
        else:
            self._current_word = ""     # normal -> spell: begin a new proper noun
        self.spell_mode = new_spell_mode

    def on_letter(self, letter: str) -> None:
        """Append a committed letter to the current spelled word (spell mode only)."""
        if self.active and self.spell_mode and letter:
            self._current_word += letter.strip().upper()

    def add_command(self, gloss: str) -> None:
        """Append a lexical command gloss (e.g. NAV, WHERE).

        Stand-in for the CTC path until it's built; the runner supplies it via a
        keypress/preset. Ignored outside an active session or during spell mode.
        """
        if self.active and not self.spell_mode and gloss:
            self.tokens.append(gloss.strip().upper())

    def on_stop(self) -> str:
        """Flush any pending word, compose the buffer, return the utterance.

        If the buffer is EMPTY (woke then stopped with no recognized signs), do NOT
        call the composer: an empty gloss list sent to the LLM makes it hallucinate
        a confused "I don't see any glosses" reply. Return "" so callers skip
        dispatch/TTS — nothing goes to ACA for an empty session.
        """
        self._flush_word()
        if not self.tokens:
            self.last_utterance = ""
            self.active = False
            return ""
        utterance = self._composer(list(self.tokens), self.wake_word)
        self.last_utterance = utterance
        self.active = False
        return utterance

    # ── internals ────────────────────────────────────────────────

    def _flush_word(self) -> None:
        if self._current_word:
            self.tokens.append(self._current_word)
            self._current_word = ""
