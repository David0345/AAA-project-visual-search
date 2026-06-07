"""Метрики ранжирования: Recall@k, Precision@k, MRR.

Все функции работают на уровне одного запроса; агрегация (micro/macro) — через
``aggregate``.  Формулы соответствуют docs/metrics-and-usage-example.md.

Инварианты:
    * ranked_ids  — упорядоченный список image_id (лучший первый);
    * target_ids  — set или list релевантных image_id;
    * если target_ids пустой — метрики = 0 (запрос пропускается при агрегации).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def recall_at_k(ranked_ids: list[int], target_ids: set[int] | list[int], k: int = 10) -> float:
    """Доля релевантных товаров, попавших в топ-k.

    Recall@k = |relevant ∩ top-k| / |relevant|

    Возвращает 0.0, если target_ids пуст.
    """
    targets = set(target_ids)
    if not targets:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in targets)
    return hits / len(targets)


def precision_at_k(ranked_ids: list[int], target_ids: set[int] | list[int], k: int = 10) -> float:
    """Доля релевантных товаров среди первых k результатов.

    Precision@k = |relevant ∩ top-k| / k

    Возвращает 0.0, если k == 0 или target_ids пуст.
    """
    if k == 0 or not target_ids:
        return 0.0
    targets = set(target_ids)
    hits = sum(1 for rid in ranked_ids[:k] if rid in targets)
    return hits / k


def mrr(ranked_ids: list[int], target_ids: set[int] | list[int]) -> float:
    """Обратный ранг первого релевантного результата.

    MRR (для одного запроса) = 1 / rank(first_relevant)

    Возвращает 0.0, если ни один релевантный результат не найден.
    """
    targets = set(target_ids)
    for i, rid in enumerate(ranked_ids):
        if rid in targets:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class ModeMetrics:
    """Агрегированные метрики для одного режима (image / txt / multimodal / all)."""

    mode: str
    count: int
    recall_at_k: dict[int, float] = field(default_factory=dict)   # k -> micro-avg
    precision_at_k: dict[int, float] = field(default_factory=dict)  # k -> micro-avg
    mrr_score: float = 0.0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_flat_dict(self) -> dict[str, float | int | str]:
        """Плоский словарь для логирования / сравнения экспериментов."""
        d: dict[str, float | int | str] = {
            "mode": self.mode,
            "count": self.count,
            "mrr": self.mrr_score,
        }
        for k, v in self.recall_at_k.items():
            d[f"recall@{k}"] = v
        for k, v in self.precision_at_k.items():
            d[f"precision@{k}"] = v
        return d


def aggregate(
    per_query_ranks: list[list[int]],
    per_query_targets: list[set[int]],
    k_values: list[int] | None = None,
    per_query_categories: list[str] | None = None,
    mode: str = "all",
) -> ModeMetrics:
    """Посчитать micro-усреднённые метрики по списку результатов поиска.

    Args:
        per_query_ranks:      ranked_ids для каждого запроса.
        per_query_targets:    target_ids (set) для каждого запроса.
        k_values:             список K (по умолчанию [1, 5, 10]).
        per_query_categories: категория (param2) для каждого запроса — нужна
                              для macro-breakdown по категориям; если None,
                              breakdown не строится.
        mode:                 название режима (для ModeMetrics.mode).

    Returns:
        ModeMetrics с micro-avg recall/precision/mrr + breakdown по категориям.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    assert len(per_query_ranks) == len(per_query_targets), (
        "ranks и targets должны иметь одинаковую длину"
    )

    rec_acc: dict[int, list[float]] = {k: [] for k in k_values}
    prec_acc: dict[int, list[float]] = {k: [] for k in k_values}
    mrr_acc: list[float] = []
    cat_acc: dict[str, dict[str, list[float]]] = {}

    for i, (ranked, targets) in enumerate(zip(per_query_ranks, per_query_targets)):
        if not targets:
            continue  # пропускаем запросы без размеченных таргетов

        mrr_val = mrr(ranked, targets)
        mrr_acc.append(mrr_val)

        for k in k_values:
            rec_acc[k].append(recall_at_k(ranked, targets, k))
            prec_acc[k].append(precision_at_k(ranked, targets, k))

        if per_query_categories is not None:
            cat = per_query_categories[i]
            if cat not in cat_acc:
                cat_acc[cat] = {
                    "mrr": [],
                    **{f"r@{k}": [] for k in k_values},
                    **{f"p@{k}": [] for k in k_values},
                }
            cat_acc[cat]["mrr"].append(mrr_val)
            for k in k_values:
                cat_acc[cat][f"r@{k}"].append(recall_at_k(ranked, targets, k))
                cat_acc[cat][f"p@{k}"].append(precision_at_k(ranked, targets, k))

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    per_category = {
        cat: {metric: _mean(vals) for metric, vals in cat_metrics.items()}
        for cat, cat_metrics in cat_acc.items()
    }

    return ModeMetrics(
        mode=mode,
        count=len(mrr_acc),
        recall_at_k={k: _mean(rec_acc[k]) for k in k_values},
        precision_at_k={k: _mean(prec_acc[k]) for k in k_values},
        mrr_score=_mean(mrr_acc),
        per_category=per_category,
    )
