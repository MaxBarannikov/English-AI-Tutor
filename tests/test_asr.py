"""ASR tests never load a Whisper model — the model boundary is stubbed."""

import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tutor import asr


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _silence(seconds: float, sample_rate: int = 16_000) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


class StubWhisper:
    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self, audio: np.ndarray, **kwargs: Any
    ) -> tuple[list[SimpleNamespace], None]:
        self.calls.append({"samples": audio.size, **kwargs})
        return [SimpleNamespace(text=text) for text in self.texts], None


def test_pcm16_is_scaled_to_unit_range() -> None:
    audio = asr.pcm16_to_float32(_pcm(0, 32767, -32768))

    assert audio.dtype == np.float32
    assert audio[0] == 0.0
    assert audio[1] == pytest.approx(1.0, abs=1e-4)
    assert audio[2] == pytest.approx(-1.0)


def test_empty_recording_yields_no_samples() -> None:
    assert asr.pcm16_to_float32(b"").size == 0


async def test_short_recording_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWhisper("should not be used")
    monkeypatch.setattr(asr, "_model", lambda: stub)

    assert await asr.transcribe(_silence(0.1)) == ""
    assert stub.calls == []


async def test_segments_are_joined_and_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWhisper(" I went ", " to Rome. ")
    monkeypatch.setattr(asr, "_model", lambda: stub)

    assert await asr.transcribe(_silence(2)) == "I went to Rome."


async def test_language_is_pinned_to_the_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubWhisper("hello")
    monkeypatch.setattr(asr, "_model", lambda: stub)

    await asr.transcribe(_silence(1))

    assert stub.calls[0]["language"] == "en"
    assert stub.calls[0]["vad_filter"] is True
