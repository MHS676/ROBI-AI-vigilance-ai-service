"""
config.py — Pydantic-Settings based configuration.
All values are read from environment variables / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Service ───────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_port: int = 8000

    # ── NestJS backend ────────────────────────────────────────────────────────
    nestjs_api_url: str = "http://localhost:3000/api/v1"
    nestjs_service_key: str = "change_me"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path: str = "mock"
    confidence_threshold: float = 0.50
    iou_threshold: float = 0.45
    alert_classes: str = "knife,gun,pistol,rifle,fight,weapon,fire,smoke"

    # ── Performance ───────────────────────────────────────────────────────────
    max_image_size: int = 1280
    workers: int = 1

    # ── Whisper audio transcription ───────────────────────────────────────────
    whisper_model_size: str = "base"     # tiny | base | small | medium | large
    whisper_max_audio_mb: int = 25       # max upload size in megabytes
    # ── Agent Monitoring ───────────────────────────────────────────────────
    # Work hours window for idle-agent alerts (24h "HH:MM" format, local time)
    work_hours_start: str   = "08:00"
    work_hours_end:   str   = "18:00"

    # Minutes the assigned agent must be absent before firing IDLE_AGENT
    idle_threshold_minutes: float = 10.0

    # Seconds two+ agents must be co-present at a table before GOSSIP_DETECTED fires
    gossip_threshold_seconds: float = 300.0   # 5 minutes

    # ArcFace cosine-distance threshold for a "match" (lower = stricter)
    face_match_threshold: float = 0.40

    # How often to snapshot agent emotion for SHI (seconds)
    shi_sample_interval_seconds: float = 30.0

    # Rolling window size for SHI average (number of samples)
    shi_window_size: int = 20

    # Minimum seconds between identity checks (reduces DeepFace overhead)
    identity_check_interval_seconds: float = 10.0
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def alert_class_set(self) -> set[str]:
        """Normalised lowercase set of class names that trigger an alert."""
        return {c.strip().lower() for c in self.alert_classes.split(",")}


@lru_cache
def get_settings() -> Settings:
    return Settings()
