"""Shared pytest configuration."""

import pytest

from llmkit import pricing


@pytest.fixture(autouse=True)
def _reset_unpriced_warnings():
    pricing._WARNED.clear()
    yield
    pricing._WARNED.clear()
