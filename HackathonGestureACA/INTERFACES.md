# Frozen Interfaces — the seams between components

> **Freeze these on day 1.** They are the contracts that let three people work in
> parallel without stepping on each other. If a seam must change, agree it together
> and update this file first, code second. See the design doc §6 for the rationale.

The pipeline is a chain of plain-data hand-offs:

```
capture ──landmarks──► recognizer ──tokens──► composer ──utterance──► aca_client ──response──► (UI)
```

Each arrow below is a fixed data shape. Depend on the shape, not on the other person's internals.

---

## 1. capture → recognizer  (owner: A)

A stream of per-frame landmark vectors.

```python
# One frame:
Landmarks = list[float]      # normalized (wrist-relative + scale). See design §21.0.
#   fingerspelling path: length 63   (MediaPipe Hands, 21 pts × xyz)
#   continuous path:     length ~150 (MediaPipe Holistic, hands+pose)

# capture yields frames continuously:
def frames() -> Iterator[Landmarks]: ...
```

---

## 2. recognizer → composer  (owner: A → B)

The completed token buffer for ONE query, in **signing order, interleaved**
(CTC glosses and fingerspelled words mixed on one timeline — see design §7).

```python
Token = str                  # a gloss "WHERE" OR a completed spelled word "TAJ MAHAL"
TokenBuffer = list[Token]    # e.g. ["WHERE", "TAJ MAHAL", "LOCATED"]

# The recognizer emits a TokenBuffer when the signer signs DONE.
```

**Person B can develop against this WITHOUT a camera** — just hand-type a `TokenBuffer`.

---

## 3. composer → aca_client  (owner: B)

One English utterance string, already wake-word-prefixed.

```python
def compose(tokens: TokenBuffer, wake_word: str = "Hey BMW") -> str: ...
#   returns e.g. "Hey BMW, where is the Taj Mahal located?"
```

---

## 4. aca_client → world  (owner: B)

Inject the utterance into real ACA, return ACA's spoken reply.

```python
def invoke_aca_live(text: str, thing_id: str, timeout_s: int = 30) -> AcaResponse: ...

class AcaResponse(TypedDict):
    success: bool
    response: str            # ACA's spoken reply text (scraped), or "" on failure
    mode: str                # "live" | "stub" | "cached"  (which fallback rung — design §16)
    error: str | None
```

Internally uses the device agent's existing `start_voice_session` + `speak_in_session`
MQTT commands. **Never** commit the device certs; read them from `/etc/iot-agent/`.

---

## 5. orchestrator ↔ ui  (owner: C)

The orchestrator drives 1–4 as a state machine (design §7 diagram) and streams state
to the UI over a WebSocket. UI is a display surface only.

```python
class UiState(TypedDict):
    mode: str                # "IDLE"|"SIGNING"|"SPELLING"|"CONFIRM"|"DISPATCH"
    tokens: TokenBuffer      # live buffer as it fills
    spelling: str            # letters collected so far in SPELL mode
    utterance: str | None    # composed sentence (shown at CONFIRM)
    aca: AcaResponse | None  # final reply
```

**Person C can develop against mocks** — a fake recognizer that emits a canned
`TokenBuffer` and a fake `invoke_aca_live` that returns a stub `AcaResponse`.

---

## Ownership summary

| Person | Owns | Mocks (to work solo) |
|---|---|---|
| **A — Vision/ML** | `capture/`, `recognizer/`, `training/`, `data/` | none (has a webcam) |
| **B — Language/Cloud** | `composer/`, `aca_client/` | hand-typed `TokenBuffer` |
| **C — Orchestration/UI** | `orchestrator/`, `ui/` | fake recognizer + fake `invoke_aca_live` |

Integration lands in `orchestrator/` (Person C), after A and B hand off tested modules.
