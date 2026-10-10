"""Shared type contracts between components. Freeze on day 1 (see INTERFACES.md).

Importing these keeps the three seams from drifting as A/B/C work in parallel.
"""
from __future__ import annotations

from typing import Iterator, Optional, TypedDict

# ── §1 capture -> recognizer ─────────────────────────────────────
Landmarks = list  # list[float]; length 63 (Hands) or ~150 (Holistic), normalized

# ── §2 recognizer -> composer ────────────────────────────────────
Token = str            # a gloss "WHERE" OR a completed spelled word "TAJ MAHAL"
TokenBuffer = list     # list[Token], signing order, interleaved


# ── §4 aca_client -> world ───────────────────────────────────────
class AcaResponse(TypedDict):
    success: bool
    response: str            # ACA's spoken reply text, or "" on failure
    mode: str                # "live" | "stub" | "cached"
    error: Optional[str]


# ── §5 orchestrator -> ui ────────────────────────────────────────
class UiState(TypedDict):
    mode: str                # "IDLE"|"SIGNING"|"SPELLING"|"CONFIRM"|"DISPATCH"
    tokens: TokenBuffer      # live buffer as it fills
    spelling: str            # letters collected so far in SPELL mode
    utterance: Optional[str]  # composed sentence (shown at CONFIRM)
    aca: Optional[AcaResponse]  # final reply
