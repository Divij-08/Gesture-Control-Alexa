# Project Status & Context — HackathonGestureACA

> Working context + implementation status for this repo. Read this first when
> picking the project back up in a new session. Full design is in
> [`docs/specs/2026-07-14-gesture-driven-aca-control-design.md`](docs/specs/2026-07-14-gesture-driven-aca-control-design.md);
> frozen component contracts are in [`INTERFACES.md`](INTERFACES.md).
>
> Last updated: 2026-07-15.

## Goal

A Deaf/mute driver controls the **real in-car ACA** (Alexa Custom Assistant) by
signing — webcam → sign language → English sentence → injected into ACA so it
*feels like signing directly invokes ACA*, not authoring a test.

> Supersedes an earlier DriveSmart-based approach (which fed gestures into a local
> Bedrock/MAYA brain via `processVoiceInput()`). This project hits **real ACA** instead.

## Locked design decisions

- **Two-stage core:** Stage 1 recognizer (landmarks → gloss tokens) + Stage 2 LLM
  composer (Bedrock/Claude: reorder ASL→English, add function words, prepend wake
  word). The LLM is both the translator AND the error-repair safety net.
- **Combined recognizer:** CTC continuous path for lexical glosses (WHERE, NAV,
  CALL…) + per-letter SPELL mode for proper nouns (which ASL always fingerspells).
  Both append to ONE shared buffer in **signing order, interleaved** (NOT
  all-CTC-then-spelling). Toggle = an explicit "SPELL" sign.
- **Recognizer models:** fingerspelling = MLP on MediaPipe Hands (63 nums/frame,
  ~98% per reference repo, borrow its confusable-set trick M/N/S, D/R/U, T/K);
  continuous = BiLSTM+CTC on MediaPipe Holistic (~150 nums/frame). Input =
  **landmarks not pixels** (robust to lighting/background/skin-tone).
- **ACA leg = `invoke_aca_live(text)`:** thin laptop MQTT client that reuses the
  device agent's EXISTING `start_voice_session` + `speak_in_session` commands (raw
  text → Polly → AudioInjectionService → ACA, scrapes reply from logcat). **No
  cloud/device redeploy, no test-case/DynamoDB/S3/LLM-judge.** Certs already at
  `/etc/iot-agent/` (laptop is provisioned as thing_id `scaletest-device-div`). Beta stage.
- **Stretch:** gloss-free SLT (Sign2GPT/GFSLT/SignLLM) run OFFLINE as a
  side-by-side "future work" benchmark, never in the live loop.

## Key TORQ finding

There is **NO text→ACA path**; ACA is audio-only. TORQ extracts the utterance
STRING from `testSteps[].action`, Polly-synthesizes it, injects audio. So "direct
invoke" = reuse `speak_in_session` (raw string) and skip the test-case carrier. The
device agent dispatches on a flat `command_type` (not test-case-shaped);
`speak_in_session` already takes raw text.

Reference code (NOT ours — read only):
- `~/Downloads/scaletest-device-agent/iot_agent/simple_agent.py` (inject_utterance
  ~L732, dispatch ~L1032, `_handle_speak_in_session` ~L1491)
- `~/workplace/torqai/src/TorqAIMCPServer/.../tools/execution.py` (TORQ MCP entry)

## Boundaries

- All our code is in this standalone repo (`~/workplace/HackathonGestureACA/`).
  **NOT a Brazil package** — we tried the code.amazon.com `create` CLI but it forces
  Brazil-package/version-set structure, so we kept it standalone git. A shareable zip
  lives at `~/workplace/HackathonGestureACA.zip` (built via `git archive`, respects `.gitignore`).
- `~/workplace/torqai/*` (team `asv-ind-gravity`) and `~/Downloads/scaletest-device-agent/`
  are **NOT ours** — reference only; never modify, commit into, or redeploy them.
- **Internal only.** Repo is full of Amazon-internal names/beta infra — keep on
  internal git, never public GitHub. Never commit device certs or credentials.
- If we ever do go the package route: bindle = ASVDevXCentral
  `amzn1.bindle.resource.oxtnpnzi5rfnadqa2vea` (team asv-ind-gravity, owner abhia).

## Collaboration (3 people)

`INTERFACES.md` freezes the 3 seams (recognizer→composer = `list[str]` tokens;
composer→aca_client = `str`; aca_client→ `AcaResponse`). `README.md` has the
3-owner split: **A** = vision/ML (`capture/`, `recognizer/`, `training/`),
**B** = language/cloud (`composer/`, `aca_client/`), **C** = orchestration/UI
(`orchestrator/`, `ui/`).

## Phased plan

P0 setup/de-risk ✅ → **P1 fingerspelling + SPELL + LLM** ✅ → P3 CTC/word lexical path
✅ (done early) → **P2 ACA leg** ⬜ (the remaining MVP piece: `invoke_aca_live` over MQTT
to real ACA) → P4 UI polish ⬜ → P5 SLT stretch ⬜. Design deliberately made the demo
**NOT depend on CTC** — and CTC/words landed anyway.

**Next up = P2, the ACA leg** (`aca_client/`): thin laptop MQTT client reusing the device
agent's `start_voice_session` + `speak_in_session`. Everything upstream (gesture → text)
is done; this connects the composed utterance to the real car.

Day-of verify: (1) laptop cert allows `iot:Publish` on the command topic; (2) use a
distinct MQTT `client_id` or IoT Core kicks the running agent off its connection.

---

## IMPLEMENTATION STATUS (2026-07-15)

**Every module in the pipeline now exists and is wired end to end.** The full flow
works: WAKE → sign command words → SPELL toggle → fingerspell a proper noun → STOP
→ compose → utterance. Shortcuts and the wake gate front it. 30 offline tests pass
(`python -m pytest tests/`); the live-camera legs are verified by hand at the webcam.

### Environment (IMPORTANT)
- **Python 3.11 or 3.12** — MediaPipe has NO wheels for 3.13/3.14 (default `python3`
  here is 3.14; use `python3.12 -m venv .venv`).
- **`mediapipe==0.10.21` pinned** — releases 0.10.30+ ship broken `py3-none` wheels
  with no compiled binary and no `mp.solutions` (which we use). 0.10.21 has the real
  compiled wheel + Holistic. Do NOT bump it. `opencv-contrib-python` comes via
  mediapipe; don't add a separate `opencv-python` (two cv2 packages fight over numpy).

### Components — all built

| Module | State | Notes |
|---|---|---|
| `capture/landmarks.py` | ✅ | Shared MediaPipe Hands extraction. 63-d single-hand (`extract_landmarks`) + 126-d both-hands (`extract_landmarks_both_hands`). `mirror_landmarks[_both]` for handedness augmentation. Extract on the RAW (unflipped) frame — train==test. |
| `capture/wake_gate.py` | ✅ | Training-free geometric control gate (MediaPipe **Holistic**). WAKE=open palm, SPELL toggle=thumb_up, STOP=thumb_left; IDLE shortcuts=shaka/peace/point. Hold-still + cooldown so signing can't false-trigger. |
| `capture/diagnose_hand.py` | ✅ | Hand-detection diagnostic (HAND/NO-HAND banner + rate). Use when a webcam script "doesn't see the hand". |
| `recognizer/fingerspell/` | ✅ | Letter MLP (63-d), `model.joblib` trained on ASL-Alphabet landmarks + mirror aug (99.5% held-out, both hands). `CONF=0.80`, `STABLE=8`. |
| `recognizer/words/` | ✅ | Word MLP (126-d both-hands), `model.joblib` = **219-class** lexical recognizer (play/music/home/weather/where/cancel/…). The design's "CTC" lexical path. `CONF=0.75`, `STABLE=12`. |
| `training/`, `training_words/` | ✅ | Image→landmark extractors + trainers for letters and words (mirror-augmented). |
| `composer/` | ✅ | `compose(tokens, wake_word)`→utterance. Bedrock Converse (Claude Haiku 4.5) with deterministic rule-based fallback; **never raises**. Env: `GESTUREACA_COMPOSER_OFFLINE`, `AWS_PROFILE`, `GESTUREACA_COMPOSER_MODEL`. Design doc: `docs/specs/2026-07-14-composer-design.md`. |
| `orchestrator/session.py` | ✅ | Pure token-buffer logic: wake/spell-toggle/letters/words/stop → interleaved TokenBuffer → compose on stop. |
| `orchestrator/wake_session.py` | ✅ | **The full live demo.** Holistic(flipped)=gate + Hands(raw)=recognizers. `--sim` scripted mode (no camera). |
| `orchestrator/shortcuts.py` | ✅ | Registry gesture→fixed utterance (shaka→music, peace→weather, point→AC). Skips the composer; fires from IDLE. |
| `orchestrator/fingerspell_to_compose.py` | ✅ | Standalone A→B demo (fingerspell word → compose). |
| `aca_client/`, `ui/`, `slt_benchmark/` | ⬜ stub | The ACA MQTT leg (P2), web UI (P4), and SLT benchmark (P5) are the remaining work. |

**Recognizer note:** the word path is an MLP classifier over static both-hands frames
(not literally BiLSTM+CTC over a sequence), but it delivers the lexical-gloss capability
the CTC path called for. Per-frame accuracy isn't perfect (e.g. music→how on a bad
frame); mitigated by the N-stable-frames gate live + the LLM composer as error-repair.

### AWS / Bedrock
Composer uses a bare `boto3.Session()` — **default profile**, nothing hardcoded. The
account is chosen by the ambient env (`AWS_PROFILE`, else default). Creds via `ada`
expire; when expired the composer auto-falls-back to rule-based (verified). Live LLM
output e.g. `Hey BMW, navigate to Cubbon Park.`; rule-based e.g. `Hey BMW, nav cubbonpark?`.

## Run sequence

From `~/workplace/HackathonGestureACA/` (Python 3.12 venv):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-fingerspell.txt   # + boto3 for the composer LLM path

# isolate each recognizer first (both draw the skeleton + show conf):
python -m recognizer.fingerspell.live_test
python -m recognizer.words.live_test
python -m capture.diagnose_hand               # if a webcam script can't see the hand

# offline sanity (no camera, no AWS):
python -m pytest tests/
GESTUREACA_COMPOSER_OFFLINE=1 python -m orchestrator.wake_session --sim --command NAV --letters CUBBONPARK

# THE FULL LIVE DEMO (refresh default AWS creds first for the LLM utterance):
python -m orchestrator.wake_session
#   IDLE: hold shaka/peace/point 1s = shortcut
#   open palm 1s = WAKE | sign command words | thumb-up = SPELL toggle |
#   fingerspell proper noun | thumb-left 1s = STOP -> compose
```

- macOS prompts for camera permission on first webcam open — allow it.
- Gate uses Holistic (weaker hand detection than Hands); `wake_session` runs it at
  `model_complexity=1`, conf 0.3, and shows a `gate: HAND/NO HAND` + hold-bar overlay.
- Models (`*.joblib`) and `data/*` are **git-ignored** (local only; models ship via the zip).
