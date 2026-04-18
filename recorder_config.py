"""
recorder_config.py — Configuration for the Falcon Synchronized Media Recorder.

All values are read from environment variables or the ai-service .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class RecorderSettings:
    # ── Storage ───────────────────────────────────────────────────────────────
    recordings_root: str = field(
        default_factory=lambda: os.getenv("RECORDINGS_ROOT", "/mnt/data/records")
    )

    # ── FFmpeg ────────────────────────────────────────────────────────────────
    ffmpeg_path: str = field(
        default_factory=lambda: os.getenv("FFMPEG_PATH", "ffmpeg")
    )

    # ── NVENC / Hardware acceleration ─────────────────────────────────────────
    # Set USE_NVENC=false to fall back to CPU libx264 (e.g. when running on
    # a machine without an NVIDIA GPU / NVENC support).
    use_nvenc: bool = field(
        default_factory=lambda: os.getenv("USE_NVENC", "true").lower() != "false"
    )

    # Segment length in seconds — one .mp4 file per segment per camera.
    # Shorter = smaller files, faster recovery after crash.
    segment_seconds: int = field(
        default_factory=lambda: int(os.getenv("SEGMENT_SECONDS", "300"))  # 5 min
    )

    # ── DVR / RTSP ────────────────────────────────────────────────────────────
    dvr_host: str = field(
        default_factory=lambda: os.getenv("DVR_HOST", "192.168.0.29")
    )
    dvr_port: int = field(
        default_factory=lambda: int(os.getenv("DVR_PORT", "554"))
    )
    dvr_user: str = field(
        default_factory=lambda: os.getenv("DVR_USER", "admin")
    )
    dvr_pass: str = field(
        default_factory=lambda: os.getenv("DVR_PASS", "Admin0123")
    )

    # ── MQTT (audio from ESP32-S3 I2S microphones) ────────────────────────────
    mqtt_host: str = field(
        default_factory=lambda: os.getenv("RABBITMQ_HOST", "localhost")
    )
    mqtt_port: int = field(
        default_factory=lambda: int(os.getenv("MQTT_PORT", "1883"))
    )
    mqtt_user: str = field(
        default_factory=lambda: os.getenv("RABBITMQ_USER", "")
    )
    mqtt_pass: str = field(
        default_factory=lambda: os.getenv("RABBITMQ_PASS", "")
    )

    # Audio topic pattern: falcon/audio/{centerId}/{tableId}/{micNumber}/pcm
    mqtt_audio_topic_pattern: str = "falcon/audio/+/+/+/pcm"

    # ── Audio (I2S / PCM from ESP32-S3) ───────────────────────────────────────
    audio_sample_rate: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    )
    audio_channels: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_CHANNELS", "1"))
    )
    # PCM bit depth from ESP32-S3 I2S (16 or 32)
    audio_bit_depth: int = field(
        default_factory=lambda: int(os.getenv("AUDIO_BIT_DEPTH", "16"))
    )

    # ── NestJS backend ────────────────────────────────────────────────────────
    nestjs_api_url: str = field(
        default_factory=lambda: os.getenv("NESTJS_API_URL", "http://localhost:4000/api/v1")
    )
    nestjs_service_key: str = field(
        default_factory=lambda: os.getenv("NESTJS_SERVICE_KEY", "change_me")
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )


@lru_cache(maxsize=1)
def get_recorder_settings() -> RecorderSettings:
    """Return a cached singleton RecorderSettings (reads .env once)."""
    # Load .env if present in the ai-service directory
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return RecorderSettings()
