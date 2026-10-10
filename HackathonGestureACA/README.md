# GestureACA

**Sign-language control of the in-car Alexa Custom Assistant (ACA).**
A webcam reads sign language, converts it to an English sentence in real time, and
injects that sentence into the real ACA via the TORQ device path — so it feels like
the signer is talking directly to the car.

> Full design: [`docs/specs/2026-07-14-gesture-driven-aca-control-design.md`](docs/specs/2026-07-14-gesture-driven-aca-control-design.md)
> Frozen component contracts: [`INTERFACES.md`](INTERFACES.md)

---

## Pipeline at a glance

```
webcam → MediaPipe → recognizer (CTC + SPELL) → token buffer
      → LLM composer (Bedrock/Claude) → invoke_aca_live() (MQTT) → real ACA → response
```

The hard, interesting part is **sign → English**; sending text to the car is a thin
MQTT wrapper over TORQ's existing `speak_in_session`. Nothing in TORQ's cloud/device
is modified — we only consume its message shape and the local device certs.

---

## Setup

```bash
git clone <internal-git-remote>/GestureACA
cd GestureACA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Prerequisites (see design §19 / Phase 0):**
- Midway auth active (`mwinit --aea`) for TORQ/Bedrock access.
- An allowlisted, provisioned, IDLE device in the farm — note its `thing_id` and wake word.
- Device certs present at `/etc/iot-agent/` (used by `aca_client`; **never** copied into the repo).
- Bedrock/Claude access confirmed from the laptop (region + model id).

---

## Repo layout

| Dir | What | Owner |
|---|---|---|
| `capture/` | webcam → MediaPipe → landmark stream | A |
| `recognizer/` | fingerspelling MLP + CTC BiLSTM + SPELL state machine + buffer | A |
| `training/` | data collection + train scripts (`collect_data.py`, `train_fingerspell.py`, `train_ctc.py`) | A |
| `data/` | recorded landmark datasets, model weights (git-ignored) | A |
| `composer/` | Bedrock/Claude: token buffer → English utterance | B |
| `aca_client/` | `invoke_aca_live()` MQTT client (reuses `speak_in_session`) | B |
| `orchestrator/` | state machine + fallback ladder; wires everything | C |
| `ui/` | FastAPI + WebSocket display (cam, tokens, sentence, reply) | C |
| `slt_benchmark/` | (stretch) offline gloss-free SLT comparison | — |
| `docs/specs/` | design doc | all |

---

## How 3 people work in parallel

Each component talks to the next through a **frozen data shape** ([`INTERFACES.md`](INTERFACES.md)),
so each person mocks the seam they depend on and integrates late:

- **A (Vision/ML)** — builds `capture` + `recognizer`; output is a `list[str]` token buffer. Tests with a webcam.
- **B (Language/Cloud)** — builds `composer` + `aca_client`; develops against a **hand-typed token list** (no camera needed).
- **C (Orchestration/UI)** — builds `orchestrator` + `ui`; develops against a **mock recognizer + mock `invoke_aca_live`**.

**Workflow:** protected `main`, feature branches per component (`feat/recognizer`, `feat/composer`, `feat/ui`), CRs to merge.

---

## Phases (see design §20)

0. Setup & de-risk → 1. Fingerspelling + SPELL + LLM → **2. ACA leg = MVP demo** →
3. CTC lexical path (the "wow") → 4. UI polish → 5. (stretch) SLT benchmark.

The demo is designed to **never depend on CTC (Phase 3) landing** — Phase 2 alone is a complete, presentable demo.

---

## ⚠️ Boundaries

- This is a **standalone laptop project**, not a Brazil package. Deps come from PyPI via `pip`.
- `~/workplace/torqai/*` (team `asv-ind-gravity`) and `~/Downloads/scaletest-device-agent/`
  are **NOT ours** — read for reference only; never modify, commit into, or redeploy them.
- **Internal only.** This repo contains Amazon-internal material — keep it on internal git,
  never public GitHub. Never commit device certs or credentials.
