"""Метрики ранжирования: Recall@k, Precision@k, MRR (micro и macro по категориям).

Формулы — в docs/metrics-and-usage-example.md. Работают на (ранжированный
список image_id) против (set таргетных image_id).

TODO(Оценка): recall_at_k / precision_at_k / mrr + агрегация.
"""

from __future__ import annotations


def recall_at_k(ranked_ids: list[int], target_ids: set[int], k: int = 10) -> float:
    raise NotImplementedError


def precision_at_k(ranked_ids: list[int], target_ids: set[int], k: int = 10) -> float:
    raise NotImplementedError


def mrr(ranked_ids: list[int], target_ids: set[int]) -> float:
    raise NotImplementedError
