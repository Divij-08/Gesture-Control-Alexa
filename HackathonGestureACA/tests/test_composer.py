"""Tests for the composer (design §8, INTERFACES.md §3).

Runs fully offline — no camera, no AWS. The Bedrock path is exercised with a
mocked converse() so we assert the request shape and the wake-word guard without
any live call. Run with:  python -m pytest tests/test_composer.py -v
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

# Ensure repo root is importable when pytest is invoked from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composer import rules, compose as compose_mod
from composer.compose import compose


# ── rule-based path (deterministic, no LLM) ──────────────────────────

def test_rule_based_lexical_glosses_lowercased():
    out = rules.compose_rule_based(["WHERE", "LOCATED"])
    assert out == "Hey BMW, where located?"


def test_rule_based_keeps_fingerspelled_proper_noun_verbatim():
    # Multi-word ALL-CAPS token is a spelled proper noun — kept as-is.
    out = rules.compose_rule_based(["NAV", "TAJ MAHAL"])
    assert "TAJ MAHAL" in out
    assert out == "Hey BMW, nav TAJ MAHAL?"


def test_rule_based_custom_wake_word():
    out = rules.compose_rule_based(["NAV", "CUBBON PARK"], wake_word="Hey Mahindra")
    assert out == "Hey Mahindra, nav CUBBON PARK?"


def test_rule_based_empty_tokens_still_valid():
    # Never return "" — a wake-word-only utterance is the floor.
    assert rules.compose_rule_based([]) == "Hey BMW."


def test_rule_based_skips_blank_tokens():
    assert rules.compose_rule_based(["WHERE", "  ", ""]) == "Hey BMW, where?"


# ── public compose(): offline flag routes to rules, never raises ─────

def test_compose_offline_uses_rules(monkeypatch):
    monkeypatch.setenv("GESTUREACA_COMPOSER_OFFLINE", "1")
    assert compose(["WHERE", "TAJ MAHAL", "LOCATED"]) == "Hey BMW, where TAJ MAHAL located?"


def test_compose_offline_never_imports_boto3(monkeypatch):
    # Offline path must work with boto3 absent. Simulate by blocking the import.
    monkeypatch.setenv("GESTUREACA_COMPOSER_OFFLINE", "1")
    with mock.patch.dict(sys.modules, {"boto3": None}):  # import boto3 -> ImportError
        assert compose(["NAV", "CUBBON PARK"]) == "Hey BMW, nav CUBBON PARK?"


def test_compose_falls_back_when_bedrock_raises(monkeypatch):
    monkeypatch.delenv("GESTUREACA_COMPOSER_OFFLINE", raising=False)
    with mock.patch("composer.bedrock.compose_with_bedrock", side_effect=RuntimeError("boom")):
        out = compose(["WHERE", "TAJ MAHAL", "LOCATED"])
    # Fell back to the deterministic path, still valid + wake-word-prefixed.
    assert out == "Hey BMW, where TAJ MAHAL located?"


# ── Bedrock path with a mocked converse (no live AWS) ────────────────

def _fake_converse_returning(text: str):
    """Build a fake bedrock-runtime client whose converse() returns `text`."""
    client = mock.MagicMock()
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": text}]}}
    }
    return client


def test_bedrock_path_used_when_online(monkeypatch):
    from composer import bedrock
    monkeypatch.delenv("GESTUREACA_COMPOSER_OFFLINE", raising=False)
    client = _fake_converse_returning("Hey BMW, where is the Taj Mahal located?")
    with mock.patch.object(bedrock, "_make_client", return_value=client):
        out = compose(["WHERE", "TAJ MAHAL", "LOCATED"])
    assert out == "Hey BMW, where is the Taj Mahal located?"
    # Verify we sent a Converse request with a system prompt and the glosses.
    _, kwargs = client.converse.call_args
    assert kwargs["system"][0]["text"].startswith("You convert American Sign Language")
    assert "TAJ MAHAL" in kwargs["messages"][0]["content"][0]["text"]
    assert kwargs["inferenceConfig"]["temperature"] == 0.0


def test_bedrock_reply_missing_wake_word_gets_prefixed(monkeypatch):
    from composer import bedrock
    monkeypatch.delenv("GESTUREACA_COMPOSER_OFFLINE", raising=False)
    # Model ignored the instruction and dropped the wake word.
    client = _fake_converse_returning("Where is the Taj Mahal located?")
    with mock.patch.object(bedrock, "_make_client", return_value=client):
        out = compose(["WHERE", "TAJ MAHAL", "LOCATED"])
    assert out == "Hey BMW, Where is the Taj Mahal located?"


def test_bedrock_multiline_reply_collapsed(monkeypatch):
    from composer import bedrock
    monkeypatch.delenv("GESTUREACA_COMPOSER_OFFLINE", raising=False)
    client = _fake_converse_returning("Hey BMW, navigate\nto Cubbon Park.")
    with mock.patch.object(bedrock, "_make_client", return_value=client):
        out = compose(["NAV", "CUBBON PARK"])
    assert "\n" not in out
    assert out == "Hey BMW, navigate to Cubbon Park."


def test_bedrock_empty_reply_falls_back(monkeypatch):
    from composer import bedrock
    monkeypatch.delenv("GESTUREACA_COMPOSER_OFFLINE", raising=False)
    client = _fake_converse_returning("")  # compose_with_bedrock raises on empty
    with mock.patch.object(bedrock, "_make_client", return_value=client):
        out = compose(["WHERE", "LOCATED"])
    assert out == "Hey BMW, where located?"  # rule-based fallback


# ── wake-word guard unit tests ───────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("where is X?", "Hey BMW, where is X?"),
    ("Hey BMW, where is X?", "Hey BMW, where is X?"),
    ("hey bmw where", "hey bmw where"),          # case-insensitive prefix match
    ("  spaced   out  ", "Hey BMW, spaced out"),
    ("", "Hey BMW."),
])
def test_ensure_wake_word(raw, expected):
    assert compose_mod._ensure_wake_word(raw, "Hey BMW") == expected
