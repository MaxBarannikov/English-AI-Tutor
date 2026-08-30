"""Configuration. The only place environment variables are read."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from tutor.models import CEFRLevel

Provider = Literal["anthropic", "openai", "ollama"]

Role = Literal["responder", "analyzer", "summary"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TUTOR_", env_file=".env", extra="ignore"
    )

    provider: Provider = "anthropic"

    responder_model: str = "claude-opus-5"
    analyzer_model: str = "claude-opus-5"
    summary_model: str = "claude-opus-5"

    ollama_base_url: str = "http://localhost:11434"

    checkpoint_db: Path = Path("sessions.db")
    default_level: CEFRLevel = "B1"

    # Optional JSON: {"<model id>": {"input": <usd per 1M>, "output": <usd per 1M>}}
    # Prices for providers not shipped in tutor.pricing belong here.
    pricing_file: Path | None = None

    def model_for(self, role: Role) -> str:
        return {
            "responder": self.responder_model,
            "analyzer": self.analyzer_model,
            "summary": self.summary_model,
        }[role]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
