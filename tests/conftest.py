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
def _no_real_arm(request, monkeypatch):
    """Fail loudly if a unit test would build a real ARM client (and shell out to az).

    Live tests are meant to reach Azure, so they are exempt.
    """
    if request.node.get_closest_marker("live"):
        return

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to build a real Arm client")

    monkeypatch.setattr(source, "Arm", refuse)
