"""Pydantic-схемы ответа сервиса поиска.

Запрос приходит как multipart/form-data (text?, image?, top_k?, mode?), поэтому
для него отдельной модели нет — поля парсит FastAPI в эндпойнте. Здесь описан
формат ответа (контракт с фронтом).
"""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    item_id: int
    score: float
    image_url: str
    title: str           # сгенерированное на трейне описание (product_text)
    param2: str | None = None
    brand: str | None = None


class SearchResponse(BaseModel):
    status: str = "success"
    query_mode: str               # "image" | "text" | "multimodal"
    results: list[SearchResult]
