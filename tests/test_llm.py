import pytest
from pydantic import ValidationError

from tutor.config import Settings, get_settings
from tutor.llm import build_chat_model


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Clients build an httpx transport at construction time; an inherited proxy
    # would make these tests depend on the developer's shell.
    for var in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)


def test_anthropic_model_without_sampling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUTOR_PROVIDER", "anthropic")
    monkeypatch.setenv("TUTOR_ANALYZER_MODEL", "claude-opus-5")
    get_settings.cache_clear()

    model = build_chat_model("analyzer")

    # Opus 5 rejects temperature with a 400 — it must not be sent.
    assert model.temperature is None
    assert model.model == "claude-opus-5"


def test_anthropic_model_that_accepts_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUTOR_PROVIDER", "anthropic")
    monkeypatch.setenv("TUTOR_ANALYZER_MODEL", "claude-haiku-4-5")
    get_settings.cache_clear()

    assert build_chat_model("analyzer").temperature == 0.0


def test_openai_temperature_is_explicit_per_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUTOR_PROVIDER", "openai")
    monkeypatch.setenv("TUTOR_RESPONDER_MODEL", "gpt-4o")
    monkeypatch.setenv("TUTOR_ANALYZER_MODEL", "gpt-4o")
    get_settings.cache_clear()

    assert build_chat_model("responder").temperature == 0.7
    assert build_chat_model("analyzer").temperature == 0.0


def test_ollama_uses_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUTOR_PROVIDER", "ollama")
    monkeypatch.setenv("TUTOR_ANALYZER_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("TUTOR_OLLAMA_BASE_URL", "http://gpu-box:11434")
    get_settings.cache_clear()

    model = build_chat_model("analyzer")

    assert model.base_url == "http://gpu-box:11434"
    assert model.temperature == 0.0


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUTOR_PROVIDER", "llamafile")

    with pytest.raises(ValidationError):
        Settings()
