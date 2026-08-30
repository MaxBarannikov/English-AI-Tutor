"""Model factory — the single place where providers are wired."""

import logging

from langchain_core.language_models import BaseChatModel

from tutor.config import Role, get_settings

logger = logging.getLogger(__name__)

# The analyzer must be reproducible; the responder needs some variety so the
# conversation does not read like a template.
_TEMPERATURE: dict[Role, float] = {"responder": 0.7, "analyzer": 0.0, "summary": 0.2}

_MAX_TOKENS: dict[Role, int] = {"responder": 1024, "analyzer": 2048, "summary": 2048}

# Claude models from Opus 4.7 onward reject temperature/top_p/top_k with a 400.
# On those, the analyzer's determinism comes from the prompt instead.
_NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def build_chat_model(role: Role) -> BaseChatModel:
    settings = get_settings()
    model = settings.model_for(role)
    temperature = _TEMPERATURE[role]
    max_tokens = _MAX_TOKENS[role]

    match settings.provider:
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            if any(model.startswith(prefix) for prefix in _NO_SAMPLING_PARAMS):
                logger.debug(
                    "%s rejects sampling parameters; temperature not sent for role %s",
                    model,
                    role,
                )
                return ChatAnthropic(model=model, max_tokens_to_sample=max_tokens)
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens,
            )

        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model, temperature=temperature, max_completion_tokens=max_tokens
            )

        case "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
                num_predict=max_tokens,
            )
