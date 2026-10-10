# Demo UI (display surface) — Design

> The live-demo UI. Component `ui/`, owner C. Shows camera + translation + final
> utterance in one window. Consumes the orchestrator's live state (the `UiState`
> spirit — INTERFACES.md §5). Last updated: 2026-07-15.

## Purpose

A single polished window for the demo showing, in real time:
- the **camera** feed with the MediaPipe hand skeleton,
- the **translation** state — mode, recognized tokens, letters being spelled,
- the **final utterance** the pipeline produced (composer output or a shortcut).

No ACA-reply panel yet (the `aca_client` MQTT leg, P2, isn't built).

## Approach: enriched OpenCV window (not browser)

**Decision:** draw a demo HUD onto the existing OpenCV window rather than build a
browser/FastAPI surface. Rationale: the camera + MediaPipe + both MLPs already run
in a blocking Python OpenCV loop (`orchestrator/wake_session.run_live`); a browser
UI would require Python to stream annotated frames + state over a WebSocket
(the camera can't move to the browser). For a live demo, **fewer moving parts =
more reliable** — no server, no WebSocket, no network. The frozen design allowed a
FastAPI/WS UI; this is a deliberate, simpler substitution for the hackathon. If a
web UI is wanted later, `draw_hud`'s `state` dict is exactly what a WS would send.

## Architecture — one pure render function

```
run_live loop (owns camera, MediaPipe, models, cv2 window)
   │  builds a plain `state` dict each frame (all values already in hand)
   ▼
ui/overlay.draw_hud(frame, state) -> frame     # PURE: no camera/model/window
   │
   ▼  cv2.imshow(frame)
```

`ui/overlay.py` exposes **one function**:

```python
def draw_hud(frame: np.ndarray, state: dict) -> np.ndarray: ...
```

Stateless and defensive (missing keys default to blank) — it cannot throw and so
cannot crash the capture loop. Being pure, it's unit-testable on a blank image
with no camera. The capture/window loop stays entirely in `wake_session`.

### The `state` dict (UiState spirit, minus `aca`)

| key | type | source in run_live |
|---|---|---|
| `mode` | `"IDLE"｜"LISTENING"｜"SPELLING"` | derived from `sess.active` + `sess.spell_mode` |
| `tokens` | `list[str]` | `sess.tokens` |
| `spelling` | `str` | `sess._current_word` |
| `utterance` | `str｜None` | last composed / shortcut utterance (persists after STOP) |
| `gesture` | `str｜None` | `gate.current_gesture` |
| `hold` | `float` 0..1 | `gate.hold_seconds / need` (control-gesture progress) |
| `hand_seen` | `bool` | Holistic saw a hand this frame |

## Layout

- **Camera** fills the window (hand skeleton already drawn by the loop).
- **Translation strip** (below camera): big color-coded `MODE` badge — grey IDLE,
  green LISTENING, amber SPELLING — then `tokens: ...` and `spelled: ...` lines.
- **Utterance banner** (bottom): the last utterance in large text; **persists**
  after STOP / a shortcut so it stays readable during the demo.
- **Gate feedback** (top corner): gesture name + hold-progress bar; red `NO HAND`
  when Holistic sees nothing (so the presenter knows detection is live).

## Error handling

`draw_hud` defaults every missing key and never raises — a display glitch must
never take down recognition. All dispatch/TTS failures are already handled upstream.

## Testing

- Unit tests: `draw_hud` on a synthetic `np.zeros((480,640,3))` for each mode →
  returns a same-shape array, doesn't raise, handles empty/None fields.
- Offline PNG renders: dump one frame per mode (IDLE / LISTENING / SPELLING /
  post-utterance) so the layout is reviewable without a camera.
- The live window with a real camera is a manual visual check.

## Out of scope
ACA-reply panel (needs P2), browser/WebSocket surface, theming/animation.
