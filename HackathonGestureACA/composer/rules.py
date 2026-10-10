"""Deterministic rule-based composer — the fallback when Bedrock is unavailable.

This path uses NO LLM and NO network: it turns a token buffer into a plausible,
wake-word-prefixed English string using fixed rules. It is intentionally dumb —
its job is to guarantee compose() always returns a usable utterance (design §16
"never die"), and to let Person B develop the whole pipeline fully offline.

It does NOT reorder ASL->English or repair recognition errors — that's the LLM's
job (design §8). When Bedrock is up, this is only a safety net.

Rules (see design §8 for what the LLM does instead):
  1. Lexical glosses (all-uppercase single words like WHERE, NAV, CALL) are
     lowercased — they're ASL glosses, not proper nouns.
  2. Fingerspelled proper nouns (multi-word or mixed-case tokens like
     "TAJ MAHAL") are kept verbatim — they are the actual name the driver spelled.
  3. Tokens are joined in signing order (no reordering).
  4. The wake word is prepended and a question mark appended, because most
     in-car queries are questions and ACA needs the wake word to trigger.
"""
from __future__ import annotations


def _is_lexical_gloss(token: str) -> bool:
    """True for a single-word ALL-CAPS gloss (WHERE, NAV) — lowercase these.

    A fingerspelled proper noun ("TAJ MAHAL", "CUBBON PARK") has a space, so it
    is NOT treated as a lexical gloss and is kept verbatim.
    """
    return token.isupper() and " " not in token.strip()


def compose_rule_based(tokens: list[str], wake_word: str = "Hey BMW") -> str:
    """Turn a token buffer into a wake-word-prefixed English string, deterministically.

    >>> compose_rule_based(["WHERE", "TAJ MAHAL", "LOCATED"])
    'Hey BMW, where TAJ MAHAL located?'
    >>> compose_rule_based(["NAV", "CUBBON PARK"], wake_word="Hey Mahindra")
    'Hey Mahindra, nav CUBBON PARK?'
    >>> compose_rule_based([])
    'Hey BMW.'
    """
    words = [t if not _is_lexical_gloss(t) else t.lower()
             for t in (tok.strip() for tok in tokens) if t]

    if not words:
        # No tokens — still return a valid wake-word utterance rather than "".
        return f"{wake_word}."

    body = " ".join(words)
    return f"{wake_word}, {body}?"
