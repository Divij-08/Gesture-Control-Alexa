"""Tests for the demo HUD overlay (ui/overlay.draw_hud).

Pure-function tests on a synthetic blank frame — no camera. Verifies draw_hud
returns a same-shape array, never raises on any mode or on missing/None fields.
    python -m pytest tests/test_overlay.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.overlay import draw_hud


def _blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


from ui.overlay import PANEL_W, letterbox_to


def test_letterbox_pads_to_target_aspect_with_white():
    # A wide canvas padded to 16:9 grows in height; padding is white; original centered.
    canvas = np.zeros((480, 1100, 3), dtype=np.uint8)
    out = letterbox_to(canvas, 1920, 1080)
    # target aspect 16:9 is TALLER than 1100:480, so height grows, width unchanged
    assert out.shape[1] == 1100
    assert abs(out.shape[0] / out.shape[1] - 1080 / 1920) < 0.02
    # top row is white padding
    assert (out[0] == 255).all()


def test_letterbox_noop_when_already_wider():
    canvas = np.zeros((480, 1100, 3), dtype=np.uint8)
    # target is wider than canvas aspect -> width would grow; still returns >= input
    out = letterbox_to(canvas, 1000, 1000)
    assert out.shape[0] >= 480 and out.shape[1] >= 1100


@pytest.mark.parametrize("mode", ["IDLE", "LISTENING", "SPELLING"])
def test_draw_hud_returns_camera_plus_panel(mode):
    frame = _blank()
    out = draw_hud(frame, {"mode": mode, "tokens": ["WHERE"], "spelling": "TAJ",
                           "utterance": "Hey BMW, where is the Taj Mahal?",
                           "gesture": "open", "hold": 0.5, "hand_seen": True})
    # split layout: a new, wider canvas (camera + right-hand panel)
    assert out.shape == (480, 640 + PANEL_W, 3)
    # the camera region is preserved unmodified on the left
    assert np.array_equal(out[:, :640], frame)


def test_draw_hud_handles_empty_state():
    # Missing everything — must default and not raise.
    out = draw_hud(_blank(), {})
    assert out.shape == (480, 640 + PANEL_W, 3)


def test_draw_hud_none_utterance_ok():
    out = draw_hud(_blank(), {"mode": "LISTENING", "utterance": None})
    assert out.shape == (480, 640 + PANEL_W, 3)


def test_draw_hud_none_frame_is_safe():
    assert draw_hud(None, {"mode": "IDLE"}) is None


def test_draw_hud_long_utterance_does_not_raise():
    # The fitted-text path must handle an over-wide string.
    long = "Hey BMW, " + "navigate to the restaurant " * 8
    out = draw_hud(_blank(), {"mode": "LISTENING", "utterance": long})
    assert out.shape == (480, 640 + PANEL_W, 3)


def test_draw_hud_actually_draws_something():
    # A blank frame should have non-zero pixels after drawing the HUD.
    out = draw_hud(_blank(), {"mode": "LISTENING", "tokens": ["PLAY"],
                              "utterance": "Hey BMW, play some music"})
    assert out.sum() > 0
