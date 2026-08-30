"""Test doubles. Nothing here touches the network."""

from collections.abc import Callable
from typing import Any

import pytest

from tutor.config import get_settings
from tutor.models import Correction, ErrorType, Severity


class StubChatModel:
    """Stands in for a chat model: returns a fixed object and records calls."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "StubChatModel":
        self.schema = schema
        return self

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append(messages)
        return self.response


@pytest.fixture
def patch_model(monkeypatch: pytest.MonkeyPatch) -> Callable[..., StubChatModel]:
    """Replace `build_chat_model` inside a node module with a stub."""

    def _patch(module: str, response: Any) -> StubChatModel:
        stub = StubChatModel(response)
        monkeypatch.setattr(f"tutor.nodes.{module}.build_chat_model", lambda role: stub)
        return stub

    return _patch


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Any:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def correction(
    severity: Severity,
    original: str = "I has",
    error_type: ErrorType = "agreement",
) -> Correction:
    return Correction(
        original=original,
        corrected="I have",
        error_type=error_type,
        severity=severity,
        explanation="Use 'have' with 'I'.",
    )
