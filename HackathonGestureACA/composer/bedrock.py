"""Bedrock Converse call — the LLM path of the composer (design §8).

Turns a token buffer into ONE natural English utterance using Claude on Amazon
Bedrock. The LLM does three jobs in one call: reorder ASL->English word order,
insert dropped function words (is/the/a/to/in), and prepend the wake word. It is
also the error-repair safety net for imperfect recognition.

CREDENTIALS / ACCOUNT: nothing is hardcoded. We build a plain boto3 Session, so
it uses the standard resolution — AWS_PROFILE if you export one, otherwise the
[default] profile. The caller chooses the account by which profile is active;
this module never names one. (Refresh the chosen profile's creds yourself, e.g.
via `ada`, before the live path works.)

This module RAISES on any failure (no creds, throttle, timeout, access denied,
malformed reply). compose.py catches everything and falls back to rules.py — so
compose() itself never throws. Keeping the raise here keeps this unit's job
single: "call the LLM, or fail loudly."
"""
from __future__ import annotations

import os

# Default to Claude Haiku 4.5 via a US cross-region inference profile. Bedrock
# requires an inference-profile ID (not a bare model ID) for the Claude 4.x
# models; the exact ID is account/region-specific and the account must have
# model access enabled. Override with GESTUREACA_COMPOSER_MODEL.
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT = (
    "You convert American Sign Language glosses from a driver into ONE natural "
    "English command or question for an in-car assistant.\n"
    "- The input is an ordered list of ASL glosses. UPPERCASE multi-word tokens "
    "are proper nouns that were fingerspelled; treat them as-is (keep their "
    "spelling, but you may capitalize them normally).\n"
    "- Reorder the glosses into natural English word order and add the function "
    "words ASL drops (is, the, a, to, in, are, do).\n"
    "- If the glosses are ambiguous or a sign was clearly misrecognized, produce "
    "the most likely in-car intent rather than a literal transcription.\n"
    "- Output EXACTLY ONE sentence and nothing else: no preamble, no quotes, no "
    "explanation. Begin it with the wake word \"{wake_word}\", followed by a comma."
)


def _make_client(region: str | None = None):
    """Build a bedrock-runtime client from the ambient AWS environment.

    boto3.Session() with no profile_name uses AWS_PROFILE if set, otherwise the
    [default] profile, and runs that profile's credential_process if it has one.
    Region resolves from the arg, then AWS_REGION / AWS_DEFAULT_REGION, then the
    profile's config.
    """
    import boto3  # imported lazily so the rule-based path needs no boto3 install

    session = boto3.Session()
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    return session.client("bedrock-runtime", region_name=region)


def compose_with_bedrock(tokens: list[str], wake_word: str = "Hey BMW") -> str:
    """Compose via Claude on Bedrock. Returns the model's sentence. RAISES on failure.

    The caller (compose.py) is responsible for catching exceptions and falling
    back. The returned string is NOT yet wake-word-guaranteed — compose.py's
    post-processing enforces the prefix in case the model drops it.
    """
    model_id = os.environ.get("GESTUREACA_COMPOSER_MODEL", DEFAULT_MODEL_ID)
    client = _make_client()

    gloss_line = " ".join(t.strip() for t in tokens if t.strip())

    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT.format(wake_word=wake_word)}],
        messages=[{"role": "user", "content": [{"text": f"Glosses: {gloss_line}"}]}],
        inferenceConfig={"maxTokens": 100, "temperature": 0.0},
    )

    # Converse shape: output.message.content is a list of blocks; take the text.
    blocks = response["output"]["message"]["content"]
    text = "".join(b.get("text", "") for b in blocks).strip()
    if not text:
        raise ValueError("Bedrock returned an empty completion")
    return text
