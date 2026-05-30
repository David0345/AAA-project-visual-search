"""Прогон val_dataset через модель+индекс -> отчёт по трём режимам.

Это логика; запускается тонкой обёрткой scripts/evaluate.py.

TODO(Оценка): evaluate(model, index) -> {mode: {recall@10, precision@10, mrr}}.
"""

from __future__ import annotations

from typing import Any


def evaluate(model: Any, index: Any) -> dict[str, dict[str, float]]:
    raise NotImplementedError
