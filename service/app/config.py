"""Конфигурация сервиса визуального поиска. Все параметры — через env (12-factor),
без констант посреди кода."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VS_", env_file=".env", extra="ignore")

    # Модель-энкодер (имя из visual_search registry) и веса дообучения
    model_name: str = "siglip2_l16_256"
    checkpoint_path: str | None = None          # *_model_only.pt; None → zero-shot

    # Артефакты индекса (готовятся офлайн скриптом индексирования)
    index_path: str = "artifacts/catalog.faiss"
    meta_path: str = "artifacts/catalog_meta.parquet"   # id -> item_id, image_path, ...
    images_base: str = "artifacts/images"               # корень превью (опц.)

    # Поиск
    top_k: int = 20
    mm_image_weight: float = 0.25               # вес картинки в multimodal-склейке
    device: str = "cpu"                          # cpu | cuda

    # Сервис
    host: str = "0.0.0.0"
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    return Settings()
