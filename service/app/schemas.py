"""Pydantic-схемы запроса/ответа сервиса поиска."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    image = "image"            # поиск по картинке
    txt = "txt"                # поиск по тексту
    multimodal = "multimodal"  # картинка + текст-модификатор


class SearchRequest(BaseModel):
    mode: SearchMode
    text: str | None = Field(default=None, description="Текстовый запрос (txt / multimodal)")
    image_b64: str | None = Field(default=None, description="Картинка-запрос в base64 (image / multimodal)")
    top_k: int | None = Field(default=None, ge=1, le=100)


class SearchHit(BaseModel):
    item_id: int
    image_path: str | None = None
    score: float


class SearchResponse(BaseModel):
    mode: SearchMode
    hits: list[SearchHit]
