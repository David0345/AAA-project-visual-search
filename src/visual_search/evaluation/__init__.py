"""evaluation — офлайн-оценка качества на val_dataset по трём режимам.

ВЛАДЕЛЕЦ: «Оценка» (Васютин Павел).

Опирается на:
    * контракт схемы val_dataset (§5.3)  → val_dataset.py / val_dataset/
    * контракт индекса (§5.4)            → index/ann.py
    * контракт энкодера (§5.2)           → models/base.py

Считает Recall@K / Precision@K / MRR для режимов image / txt / multimodal.
Использует тот же ANN-индекс, что и сервис — метрики соответствуют проду.

Пример::

    from visual_search.evaluation.evaluate import evaluate, print_report
    results = evaluate(model, index, images_base="data/raw/dataset_1M")
    print_report(results)
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
