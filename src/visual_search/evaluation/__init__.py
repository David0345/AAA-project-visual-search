"""evaluation — офлайн-оценка качества на val_dataset по трём режимам.

Считает Recall@K / Precision@K / MRR для режимов image / txt / multimodal.
Использует тот же ANN-индекс, что и сервис — метрики соответствуют проду.
"""

from .metrics import ModeMetrics, aggregate, mrr, precision_at_k, recall_at_k
from .val_dataset import ValDataset, ValQuery
from .evaluate import evaluate, evaluate_with_search_fn, print_report

__all__ = [
    # metrics
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "aggregate",
    "ModeMetrics",
    # val_dataset
    "ValDataset",
    "ValQuery",
    # evaluate
    "evaluate",
    "evaluate_with_search_fn",
    "print_report",
]
