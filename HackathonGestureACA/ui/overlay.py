"""Demo HUD overlay — split layout: camera on the left, WHITE text panel on the right.

ONE pure function: draw_hud(frame, state) -> canvas. It builds a NEW wider canvas
with the camera frame on the left and a white text panel on the right (mode, live
translation, final utterance, gate feedback). No camera, no models, no cv2 window —
the orchestrator (orchestrator/wake_session.run_live) owns the capture loop and
imshows whatever this returns. See docs/specs/2026-07-15-demo-ui-design.md.

Color scheme: white panel, BLACK text (readable on white). Green / amber / yellow /
red appear only as fills, bars, and accents (status indicators) — never as text on
white, where they'd wash out.

Returns a new array (camera + panel), NOT the input frame — the caller displays the
return value. Defensive: every field defaults, so a display glitch can't crash the loop.
"""
from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR — text is black; colors are indicators only
_PANEL_BG = (255, 255, 255)   # white
_INK = (24, 24, 24)           # near-black text
_DIM = (130, 130, 130)        # grey labels
_LINE = (210, 210, 210)       # dividers
_CARD_BG = (238, 238, 238)    # light-grey utterance card

_GREEN = (0, 160, 0)
_AMBER = (0, 170, 235)
_YELLOW = (0, 200, 255)
_RED = (40, 40, 210)
_IDLE_GREY = (200, 200, 200)

# MODE badge fill colors (badge text is always black)
_MODE_FILL = {"IDLE": _IDLE_GREY, "LISTENING": _GREEN, "SPELLING": _AMBER}

PANEL_W = 460
_PAD = 24


def _wrap(text, max_w, scale, thickness):
    """Greedy word-wrap `text` into lines that fit within max_w pixels."""
    words = text.split()
    lines, cur = [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        (tw, _), _ = cv2.getTextSize(trial, _FONT, scale, thickness)
        if tw <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _text(img, s, x, y, scale, color, thickness=1):
    cv2.putText(img, s, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, float(frac)))
    cv2.rectangle(img, (x, y), (x + w, y + h), _LINE, 1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


def draw_hud(frame, state: dict):
    """Compose [camera | white text-panel] and return the combined canvas (BGR ndarray).

    `state` keys (all optional): mode, tokens, spelling, utterance, gesture,
    hold (0..1), hand_seen.
    """
    if frame is None:
        return frame
    ch, cw = frame.shape[:2]

    mode = state.get("mode", "IDLE")
    tokens = state.get("tokens") or []
    spelling = state.get("spelling") or ""
    utterance = state.get("utterance")
    gesture = state.get("gesture")
    hold = state.get("hold", 0.0) or 0.0
    hand_seen = state.get("hand_seen", False)
    badge_fill = _MODE_FILL.get(mode, _IDLE_GREY)

    canvas = np.zeros((ch, cw + PANEL_W, 3), dtype=np.uint8)
    canvas[:, :cw] = frame
    panel = canvas[:, cw:]
    panel[:] = _PANEL_BG
    px = _PAD
    inner_w = PANEL_W - 2 * _PAD

    # ── title ──
    _text(panel, "GestureACA", px, 46, 1.0, _INK, 2)
    _text(panel, "sign -> speech", px, 70, 0.5, _DIM, 1)
    cv2.line(panel, (px, 86), (PANEL_W - px, 86), _LINE, 1)

    # ── MODE badge (colored fill, black text) + HAND status ──
    by = 106
    (mw, _mh), _ = cv2.getTextSize(mode, _FONT, 0.8, 2)
    cv2.rectangle(panel, (px, by), (px + mw + 24, by + 40), badge_fill, -1)
    _text(panel, mode, px + 12, by + 28, 0.8, (20, 20, 20), 2)
    hand_txt = "HAND" if hand_seen else "NO HAND"
    hand_col = _GREEN if hand_seen else _RED
    (hw, _), _ = cv2.getTextSize(hand_txt, _FONT, 0.6, 2)
    _text(panel, hand_txt, PANEL_W - px - hw, by + 28, 0.6, hand_col, 2)

    # ── gesture + hold bar ──
    gy = by + 74
    if gesture:
        _text(panel, f"gesture: {gesture}", px, gy, 0.55, _DIM, 1)
        _bar(panel, px, gy + 12, inner_w, 12, hold, badge_fill)

    # ── TRANSLATION section (black text) ──
    ty = gy + 44
    _text(panel, "TRANSLATION", px, ty, 0.55, _DIM, 1)
    ty += 26
    _text(panel, "tokens", px, ty, 0.5, _DIM, 1)
    for ln in _wrap(" ".join(tokens) if tokens else "-", inner_w, 0.7, 2)[:2]:
        ty += 26
        _text(panel, ln, px, ty, 0.7, _INK, 2)
    ty += 32
    _text(panel, "spelling", px, ty, 0.5, _DIM, 1)
    ty += 26
    _text(panel, spelling or "-", px, ty, 0.7, _INK, 2)

    # ── UTTERANCE card (bottom): light-grey card, yellow accent, black text ──
    card_top = ch - 132
    cv2.rectangle(panel, (px - 8, card_top), (PANEL_W - px + 8, ch - 20), _CARD_BG, -1)
    cv2.rectangle(panel, (px - 8, card_top), (PANEL_W - px + 8, card_top + 4), _YELLOW, -1)
    _text(panel, "UTTERANCE -> ACA", px, card_top + 30, 0.5, _DIM, 1)
    if utterance:
        uy = card_top + 62
        for ln in _wrap(utterance, inner_w, 0.65, 2)[:3]:
            _text(panel, ln, px, uy, 0.65, _INK, 2)
            uy += 28
    else:
        _text(panel, "(waiting...)", px, card_top + 70, 0.7, _DIM, 2)

    return canvas


def letterbox_to(canvas, target_w, target_h, fill=_PANEL_BG):
    """Pad `canvas` (not scale) to the target aspect ratio, centered, with `fill`.

    In fullscreen, OpenCV fits the image to the screen and pads the leftover with
    the WINDOW's black background — giving ugly black bars above/below our wide
    canvas. Padding here to the screen's aspect ratio with WHITE instead means the
    bars blend into the panel. Aspect-only: we add margin, we don't rescale, so the
    OS fullscreen scale stays uniform and text isn't distorted.
    """
    h, w = canvas.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return canvas
    # grow whichever dimension is short of the target aspect
    if w / h < target_w / target_h:
        new_w = int(round(h * target_w / target_h))
        new_h = h
    else:
        new_w = w
        new_h = int(round(w * target_h / target_w))
    if new_w <= w and new_h <= h:
        return canvas
    out = np.empty((max(new_h, h), max(new_w, w), 3), dtype=canvas.dtype)
    out[:] = fill
    y0 = (out.shape[0] - h) // 2
    x0 = (out.shape[1] - w) // 2
    out[y0:y0 + h, x0:x0 + w] = canvas
    return out


__all__ = ["draw_hud", "letterbox_to", "PANEL_W"]
