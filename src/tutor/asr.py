"""Speech recognition for voice input.

Runs locally with faster-whisper: no API key, no per-minute cost, and the
learner's audio never leaves the machine. The model is loaded once, lazily, and
transcription runs in a worker thread because it is CPU-bound.
"""

import asyncio
import logging
from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray

from tutor.config import get_settings

logger = logging.getLogger(__name__)

_INT16_FULL_SCALE = 32768.0

# Below this, the recording is a stray click rather than speech.
MIN_DURATION_SECONDS = 0.4


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    settings = get_settings()
    logger.info(
        "loading ASR model %s (%s, %s) — first run downloads it",
        settings.asr_model,
        settings.asr_device,
        settings.asr_compute_type,
    )
    return WhisperModel(
        settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )


def pcm16_to_float32(raw: bytes) -> NDArray[np.float32]:
    """Chainlit streams mono signed 16-bit PCM; Whisper wants float32 in [-1, 1]."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples / _INT16_FULL_SCALE


def _transcribe_sync(audio: NDArray[np.float32]) -> str:
    settings = get_settings()
    segments, _ = _model().transcribe(
        audio,
        language=settings.asr_language,
        beam_size=5,
        vad_filter=True,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


async def transcribe(raw_pcm: bytes) -> str:
    """Transcribe one recording. Returns "" when there is nothing to hear."""
    audio = pcm16_to_float32(raw_pcm)
    duration = audio.size / get_settings().asr_sample_rate
    if duration < MIN_DURATION_SECONDS:
        logger.info("recording too short to transcribe: %.2fs", duration)
        return ""

    text = await asyncio.to_thread(_transcribe_sync, audio)
    logger.info("transcribed %.1fs of audio into %d characters", duration, len(text))
    return text


if __name__ == "__main__":
    # Downloads the model so the first dictation is not a minute-long wait.
    logging.basicConfig(level=logging.INFO)
    _model()
    print(f"ASR model {get_settings().asr_model} ready")
