# Composer (Stage 2) — Design

> The LLM composer: token buffer → one English utterance. Component `composer/`,
> owner B. Implements design doc §8 and the frozen seam in
> [`INTERFACES.md`](../../INTERFACES.md) §3. Last updated: 2026-07-14.

## Purpose

Stage 2 of the two-stage core. Turns the recognizer's token buffer (ASL glosses,
in signing order) into ONE natural English command/question, wake-word-prefixed,
ready for `aca_client.invoke_aca_live()`.

```
["WHERE", "TAJ MAHAL", "LOCATED"]  ──compose()──►  "Hey BMW, where is the Taj Mahal located?"
```

The LLM does three jobs in one call (design §8): reorder ASL→English word order,
insert dropped function words (is/the/a/to/in), prepend the wake word. It is also
the **error-repair safety net** — a dropped or misread gloss still yields a
coherent sentence rather than garbage to the car.

## Frozen contract (INTERFACES.md §3 — do not change unilaterally)

```python
def compose(tokens: list[str], wake_word: str = "Hey BMW") -> str: ...
```

`compose()` **never raises** and **never returns `""`**. Every path ends in the
wake-word guard, so a misfire cannot send an untriggerable or empty utterance to
ACA (design §16 "never die").

## Architecture — two paths, one guard

```
compose(tokens, wake_word)
   │
   ├─ GESTUREACA_COMPOSER_OFFLINE set ───────────────┐
   ├─ try Bedrock Converse (Claude Haiku) ─ success ─┤
   └─ Bedrock raises (no creds / throttle / error) ──┤
                                                       ▼
                        ┌───────────────────────────────────────────┐
                        │ _ensure_wake_word: force wake-word prefix,  │
                        │ collapse to one line → return str           │
                        └───────────────────────────────────────────┘
                             (fallback body = rule-based composer)
```

| Module | Responsibility | Fails how |
|---|---|---|
| [`composer/compose.py`](../../composer/compose.py) | Public `compose()`: fallback ladder + wake-word guard + CLI | Never raises |
| [`composer/bedrock.py`](../../composer/bedrock.py) | boto3 Converse call to Claude on Bedrock + the §8 system prompt | **Raises** on any failure (caught by compose) |
| [`composer/rules.py`](../../composer/rules.py) | Deterministic fallback: lowercase lexical glosses, keep fingerspelled proper nouns verbatim, prepend wake word, append `?` | Pure/total |

Each unit has one job; the "raise vs. never-raise" split lives at the
`compose.py` boundary. Lazy `import boto3` inside the Bedrock path means the
offline path needs no boto3 installed.

## LLM call

Amazon Bedrock **Converse API**, Claude **Haiku 4.5** (tiny task — ~10 tokens →
one sentence, once per query; latency and cost favor Haiku). `temperature=0`,
`maxTokens=100`. System prompt = the design §8 sketch; user message = the gloss
list. Reply text extracted from `output.message.content[].text`.

## Fallback rules (rule-based path)

Not a translator — a safety net. It does NOT reorder or repair.
- Single-word ALL-CAPS token → lexical gloss → lowercased (`WHERE` → `where`).
- Multi-word / spaced ALL-CAPS token → fingerspelled proper noun → kept verbatim
  (`TAJ MAHAL` stays `TAJ MAHAL`).
- Join in signing order, prepend `wake_word,`, append `?`. Empty input →
  `"{wake_word}."` (never `""`).

## Configuration (all environment; nothing hardcoded)

| Variable | Effect | Default |
|---|---|---|
| `GESTUREACA_COMPOSER_OFFLINE` | Truthy → skip LLM, rule-based only (offline dev) | unset (LLM on) |
| `AWS_PROFILE` / `AWS_REGION` | Chooses the AWS account/region for Bedrock | **default profile** (per decision) |
| `GESTUREACA_COMPOSER_MODEL` | Override the Bedrock model/inference-profile ID | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `GESTUREACA_WAKE_WORD` | Wake word for the CLI | `Hey BMW` |

**Credentials / account decision.** `bedrock.py` builds a plain `boto3.Session()`
— no `profile_name`, no account named in code. boto3 resolves the profile from
the ambient environment (`AWS_PROFILE`, else `[default]`) and runs that profile's
credential process. The operator chooses and refreshes the account; the composer
never selects one. This satisfies the production-safety rule against picking a
billing account on the user's behalf.

## Testing

[`tests/test_composer.py`](../../tests/test_composer.py) — 17 tests, **fully
offline**. Rule-based path asserted directly (deterministic); Bedrock path tested
with a mocked `converse()` (asserts request shape + wake-word guard); fallback
ladder tested by making the Bedrock call raise. No camera, no live AWS.

Develop against a hand-typed `TokenBuffer` — Person B needs neither.

## Verification (2026-07-14)

- 17/17 tests pass.
- Offline CLI: `GESTUREACA_COMPOSER_OFFLINE=1 python -m composer.compose WHERE "TAJ MAHAL" LOCATED`
  → `Hey BMW, where TAJ MAHAL located?`
- Live path with expired default creds caught `ExpiredTokenException`, logged it,
  and fell back to rule-based — the §16 "never die" behavior, confirmed.

## Known caveats / follow-ups

- The Haiku 4.5 inference-profile ID is the assumed US cross-region form; not yet
  confirmed against a live account (default creds were expired at build time).
  Env-overridable; verify the exact ID + Bedrock model access once creds are fresh.
- Rule-based output is deliberately un-grammatical (no reordering). It exists to
  keep the demo alive, not to impress — the LLM path is the real Stage 2.
