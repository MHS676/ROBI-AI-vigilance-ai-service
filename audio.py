"""
audio.py — Whisper-powered audio transcription for Falcon Security.
====================================================================

Loaded as a singleton at startup alongside the YOLO model.

Supports audio sent from:
  • NestJS backend  (already stored on disk or passed as raw bytes)
  • ESP32 AI-Mic    (WAV / PCM buffer forwarded through NestJS)

Accepted formats: wav, mp3, m4a, ogg, flac, webm
Maximum file size is enforced via WHISPER_MAX_AUDIO_MB env var (default 25 MB).

Thread safety:
  Whisper's transcribe() is synchronous and not thread-safe when called
  concurrently with the same model instance.  We use asyncio.Lock() so
  concurrent requests queue rather than corrupt model state.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from typing import Optional

from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Whisper model singleton
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None
_whisper_lock  = asyncio.Lock()

# Formats Whisper / ffmpeg can reliably handle
SUPPORTED_AUDIO_TYPES: set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",           # mp3
    "audio/mp4",            # m4a
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/x-m4a",
    "application/octet-stream",  # raw bytes — accepted, format inferred by ffmpeg
}

SUPPORTED_EXTENSIONS: set[str] = {
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4",
}

MAX_AUDIO_BYTES: int = int(os.getenv("WHISPER_MAX_AUDIO_MB", "25")) * 1024 * 1024
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")


# ─────────────────────────────────────────────────────────────────────────────
# Loader  (called once from main.py lifespan)
# ─────────────────────────────────────────────────────────────────────────────

def load_whisper_model():
    """
    Load the Whisper model into memory and return it.
    Returns None if the `openai-whisper` package is not installed, so the
    rest of the service continues to function without transcription support.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper  # type: ignore

        logger.info(f"🎙️  Loading Whisper model '{WHISPER_MODEL_SIZE}'…")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.success(f"✅ Whisper '{WHISPER_MODEL_SIZE}' ready")
    except ImportError:
        logger.warning(
            "⚠️  openai-whisper is not installed — POST /transcribe-audio will return 503.\n"
            "    Install with: pip install openai-whisper"
        )
        _whisper_model = None
    except Exception as exc:
        logger.error(f"❌ Whisper failed to load: {exc}")
        _whisper_model = None

    return _whisper_model


def get_whisper_model():
    """Return the already-loaded Whisper model (or None if unavailable)."""
    return _whisper_model


# ─────────────────────────────────────────────────────────────────────────────
# Transcription helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_confidence(whisper_result: dict) -> float:
    """
    Whisper doesn't expose a top-level confidence score, but each segment
    carries avg_logprob (log probability).  We exponentiate the mean across
    all segments to get a 0-1 pseudo-confidence.

    Falls back to 0.0 if segments are missing (e.g. silent / empty audio).
    """
    segments = whisper_result.get("segments", [])
    if not segments:
        return 0.0

    avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
    # avg_logprob is typically in [-1, 0]; clamp to [0, 1]
    confidence = float(min(max(pow(2.718281828, avg_logprob), 0.0), 1.0))
    return round(confidence, 4)


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe raw audio bytes using Whisper.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio content read from the uploaded file.
    filename : str
        Original filename — used to infer extension for the temp file.
    language : str, optional
        ISO-639-1 language hint (e.g. 'en', 'ar').  When None, Whisper
        auto-detects the language.

    Returns
    -------
    dict with keys: transcribed_text, detected_language, confidence_score,
                    duration_s, inference_ms, model_size
    """
    model = get_whisper_model()
    if model is None:
        raise RuntimeError("Whisper model is not loaded")

    # Derive a safe extension for the temp file so ffmpeg decodes correctly
    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    if ext not in SUPPORTED_EXTENSIONS:
        ext = ".wav"

    t0 = time.perf_counter()

    # Write to a named temp file — Whisper requires a file path, not a buffer
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        async with _whisper_lock:
            # Run the blocking transcribe() in the default thread pool so it
            # doesn't block the FastAPI event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # default ThreadPoolExecutor
                lambda: model.transcribe(
                    tmp_path,
                    language=language,
                    fp16=False,          # fp16 requires CUDA; safe default is False
                    verbose=False,
                ),
            )
    finally:
        # Always remove the temp file, even on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    transcribed_text  = (result.get("text") or "").strip()
    detected_language = result.get("language") or "unknown"
    confidence_score  = _estimate_confidence(result)

    # Duration: sum of segment end times, or 0 if unavailable
    segments  = result.get("segments", [])
    duration_s = round(segments[-1]["end"], 2) if segments else 0.0

    logger.info(
        f"🎙️  Transcription complete | lang={detected_language} "
        f"confidence={confidence_score:.2f} dur={duration_s}s "
        f"inference={inference_ms}ms"
    )

    return {
        "transcribed_text":  transcribed_text,
        "detected_language": detected_language,
        "confidence_score":  confidence_score,
        "duration_s":        duration_s,
        "inference_ms":      inference_ms,
        "model_size":        WHISPER_MODEL_SIZE,
    }
