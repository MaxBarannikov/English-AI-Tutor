.DEFAULT_GOAL := help
.PHONY: help install run test lint format typecheck check evals asr clean

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

run:  ## Run the Chainlit UI
	uv run chainlit run app.py

test:  ## Run the test suite (no network)
	uv run pytest

lint:  ## Lint and autofix
	uv run ruff check --fix .

format:  ## Format
	uv run ruff format .

typecheck:  ## Type-check with mypy strict
	uv run mypy src

check: lint typecheck test  ## Everything CI would run
	uv run ruff format --check .

evals:  ## Score the analyzer against the labelled dataset (calls a real model)
	uv run python evals/run.py

asr:  ## Download the speech-recognition model ahead of first use
	uv run python -m tutor.asr

clean:  ## Remove caches and the local session database
	rm -rf .pytest_cache .mypy_cache .ruff_cache .chainlit
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f sessions.db sessions.db-shm sessions.db-wal
