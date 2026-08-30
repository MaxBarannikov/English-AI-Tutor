"""Prompts live as .md files so they stay diffable and reviewable."""

from importlib.resources import files

_DIR = files(__name__)


def _load(name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


RESPONDER = _load("responder")
ANALYZER = _load("analyzer")
SUMMARY = _load("summary")
