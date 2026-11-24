"""Shared pytest configuration."""

import pytest

from llmkit import pricing
from llmkit.azure import source, target


@pytest.fixture(autouse=True)
def _reset_unpriced_warnings():
    pricing._WARNED.clear()
    yield
    pricing._WARNED.clear()


@pytest.fixture(autouse=True)
def _reset_routing_warnings():
    target._WARNED.clear()
    yield
    target._WARNED.clear()


@pytest.fixture(autouse=True)
def _no_real_arm(monkeypatch):
    """Fail loudly if a test would build a real ARM client (and shell out to az)."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to build a real Arm client")

    monkeypatch.setattr(source, "Arm", refuse)
