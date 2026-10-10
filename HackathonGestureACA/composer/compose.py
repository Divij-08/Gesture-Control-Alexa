"""Public composer entrypoint — the frozen seam recognizer -> aca_client (INTERFACES.md §3).

    compose(tokens: list[str], wake_word: str = "Hey BMW") -> str

Stage 2 of the two-stage core (design §8): token buffer -> ONE English utterance,
wake-word-prefixed, ready to hand to aca_client.invoke_aca_live().

Fallback ladder (design §16 "never die"):
  1. If GESTUREACA_COMPOSER_OFFLINE is set, skip the LLM entirely (offline dev).
  2. Otherwise try the Bedrock/Claude path (bedrock.compose_with_bedrock).
  3. On ANY failure — no boto3, no creds, throttle, timeout, bad reply — fall
     back to the deterministic rule-based composer (rules.compose_rule_based).

compose() NEVER raises and NEVER returns "": every path ends in the wake-word
guard, so a misfire can't send garbage (or nothing) to the car.

Develop against a hand-typed token list; no camera and no AWS needed
(set GESTUREACA_COMPOSER_OFFLINE=1).
"""
from __future__ import annotations

import os

from composer.rules import compose_rule_based


def _offline() -> bool:
    """True when the LLM path should be skipped (env flag set to a truthy value)."""
    return os.environ.get("GESTUREACA_COMPOSER_OFFLINE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ensure_wake_word(sentence: str, wake_word: str) -> str:
    """Guarantee the utterance starts with the wake word and is a single line.

    ACA only triggers on the wake word, so if the LLM dropped or altered it we
    prepend it. Also collapse to one line so a stray newline can't split the
    injected utterance.
    """
    sentence = " ".join(sentence.split()).strip()  # collapse whitespace/newlines
    if not sentence:
        return f"{wake_word}."
    # Case-insensitive check so "hey bmw" from the model still counts as prefixed.
    if not sentence.lower().startswith(wake_word.lower()):
        sentence = f"{wake_word}, {sentence}"
    return sentence


def compose(tokens: list[str], wake_word: str = "Hey BMW") -> str:
    """Token buffer -> wake-word-prefixed English utterance. Never raises.

    >>> import os; os.environ["GESTUREACA_COMPOSER_OFFLINE"] = "1"
    >>> compose(["WHERE", "TAJ MAHAL", "LOCATED"])
    'Hey BMW, where TAJ MAHAL located?'
    """
    # Never send an empty gloss list to the LLM — with nothing to translate it
    # hallucinates a confused "I don't see any glosses" reply. Use the
    # deterministic rule-based path (which returns just the wake word). This is a
    # defense-in-depth guard; callers (orchestrator.Session) should skip compose()
    # entirely on an empty buffer.
    if not [t for t in tokens if t and t.strip()]:
        return _ensure_wake_word(compose_rule_based(tokens, wake_word), wake_word)

    if not _offline():
        try:
            # Imported lazily: the offline path must not require boto3 to be installed.
            from composer.bedrock import compose_with_bedrock
            return _ensure_wake_word(compose_with_bedrock(tokens, wake_word), wake_word)
        except Exception as e:  # noqa: BLE001 — deliberate: any failure -> safe fallback
            print(f"[composer] Bedrock path failed ({type(e).__name__}: {e}); "
                  f"falling back to rule-based.")

    return _ensure_wake_word(compose_rule_based(tokens, wake_word), wake_word)


def _main() -> None:
    """CLI: compose an utterance from glosses passed as argv. For live/offline smoke tests.

        python -m composer.compose WHERE "TAJ MAHAL" LOCATED
        GESTUREACA_COMPOSER_OFFLINE=1 python -m composer.compose NAV "CUBBON PARK"
    """
    import sys

    tokens = sys.argv[1:]
    if not tokens:
        print('usage: python -m composer.compose GLOSS ["MULTI WORD PROPER NOUN"] ...')
        raise SystemExit(2)
    wake_word = os.environ.get("GESTUREACA_WAKE_WORD", "Hey BMW")
    print(compose(tokens, wake_word))


if __name__ == "__main__":
    _main()
