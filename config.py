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
