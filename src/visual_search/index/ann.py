"""Обёртка над ANN-бэкендом (faiss / hnswlib) — единый интерфейс поиска.

Бэкенд и стратегия квантования — точка тюнинга под < 1 сек / 1 rps; интерфейс
search(query_vec, k) -> [(image_id, score), ...] от этого не зависит.

TODO(index): build / save / load / search.
"""

from __future__ import annotations

from visual_search.common.logging import get_logger

logger = get_logger(__name__)
