"""Прогон val_dataset через модель + ANN-индекс → отчёт по трём режимам.

Это логика пайплайна; запускается тонкой обёрткой ``scripts/evaluate.py``.

Основная точка входа::

    results = evaluate(model, index, dataset_path="...", images_base="...")
    # results["txt"]["recall@10"]  → 0.73
    # results["all"]["mrr"]        → 0.61

Дополнительная точка входа для случаев, когда кодирование управляется снаружи::

    results = evaluate_with_search_fn(
        search_fn=lambda q: index.search(model.encode_query(q), k=10),
        dataset_path="...",
    )

Интерфейсы, от которых зависит этот модуль:
    * ``Encoder``  — ``models/base.py``:
        encode_image(images: Tensor) → Tensor (B, D)  [L2-norm]
        encode_text(tokens: Tensor)  → Tensor (B, D)  [L2-norm]
    * ``ANN``      — ``index/ann.py``:
        search(query_vec: np.ndarray, k: int) → list[tuple[int, float]]
        ([(image_id, score), ...], лучший первый)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .metrics import ModeMetrics, aggregate
from .val_dataset import ValDataset, ValQuery

logger = logging.getLogger(__name__)

MODES = ("image", "txt", "multimodal")
DEFAULT_K = 10
DEFAULT_K_VALUES = [1, 5, 10]


# ---------------------------------------------------------------------------
# Вспомогательные функции кодирования запроса
# ---------------------------------------------------------------------------


def _encode_query(query: ValQuery, model, tokenizer, image_transform) -> np.ndarray:
    """Закодировать один запрос в вектор через модель.

    Логика по режимам:
        image       → encode_image
        txt         → encode_text
        multimodal  → среднее (encode_image + encode_text), L2-нормировано

    Args:
        query:            ValQuery из ValDataset.
        model:            объект, реализующий Encoder (models/base.py).
        tokenizer:        callable(text: str) → Tensor(1, L)
        image_transform:  callable(PIL.Image) → Tensor(1, C, H, W)

    Returns:
        np.ndarray shape (D,) — L2-нормированный вектор.
    """
    import torch
    from PIL import Image

    vecs: list[np.ndarray] = []

    if query.image_path is not None and query.mode in ("image", "multimodal"):
        img = Image.open(query.image_path).convert("RGB")
        img_tensor = image_transform(img).unsqueeze(0)  # (1, C, H, W)
        with torch.no_grad():
            img_vec = model.encode_image(img_tensor).squeeze(0).cpu().numpy()
        vecs.append(img_vec)

    if query.txt_query is not None and query.mode in ("txt", "multimodal"):
        tokens = tokenizer(query.txt_query)  # (1, L)
        with torch.no_grad():
            txt_vec = model.encode_text(tokens).squeeze(0).cpu().numpy()
        vecs.append(txt_vec)

    if not vecs:
        raise ValueError(
            f"query_id={query.query_id}: нет ни image_path, ни txt_query для mode={query.mode!r}"
        )

    vec = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Главные точки входа
# ---------------------------------------------------------------------------


def evaluate_with_search_fn(
    search_fn: Callable[[ValQuery], list[int]],
    dataset_path: str | None = None,
    images_base: str = "",
    k_values: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, ModeMetrics]:
    """Оценить качество поиска, используя произвольную функцию поиска.

    Это низкоуровневая функция — полезна для тестирования и быстрых экспериментов.

    Args:
        search_fn:    callable(ValQuery) → list[int]
                      Принимает запрос, возвращает ranked список image_id.
        dataset_path: путь к val_dataset.csv; если None — используется путь по умолчанию.
        images_base:  префикс путей к изображениям.
        k_values:     список K для Recall@K и Precision@K (по умолчанию [1, 5, 10]).
        verbose:      логировать прогресс.

    Returns:
        Словарь ``{mode: ModeMetrics}`` для каждого режима + ключ ``"all"``.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    ds_kwargs: dict = {} if dataset_path is None else {"csv_path": dataset_path}
    dataset = ValDataset(**ds_kwargs, images_base=images_base)

    if verbose:
        st = dataset.stats()
        logger.info(
            "ValDataset loaded: total=%d  image=%d  txt=%d  multimodal=%d",
            st["total"], st["image"], st["txt"], st["multimodal"],
        )

    results: dict[str, ModeMetrics] = {}
    all_ranks: list[list[int]] = []
    all_targets: list[set[int]] = []
    all_categories: list[str] = []

    for mode in MODES:
        queries = dataset.get_by_mode(mode)
        if not queries:
            if verbose:
                logger.warning("Режим %r: нет запросов, пропускаем.", mode)
            continue

        ranks: list[list[int]] = []
        targets: list[set[int]] = []
        categories: list[str] = []

        for q in queries:
            ranked = search_fn(q)
            ranks.append(ranked)
            targets.append(q.target_image_ids)
            categories.append(str(q.metadata.get("param2") or "unknown"))

        mode_metrics = aggregate(ranks, targets, k_values, categories, mode=mode)
        results[mode] = mode_metrics

        if verbose:
            flat = mode_metrics.as_flat_dict()
            logger.info(
                "[%s] n=%d  recall@%d=%.3f  precision@%d=%.3f  mrr=%.3f",
                mode, flat["count"],
                DEFAULT_K, flat.get(f"recall@{DEFAULT_K}", 0),
                DEFAULT_K, flat.get(f"precision@{DEFAULT_K}", 0),
                flat["mrr"],
            )

        all_ranks.extend(ranks)
        all_targets.extend(targets)
        all_categories.extend(categories)

    # Агрегат по всем режимам
    results["all"] = aggregate(all_ranks, all_targets, k_values, all_categories, mode="all")

    if verbose:
        flat = results["all"].as_flat_dict()
        logger.info(
            "[all]  n=%d  recall@%d=%.3f  precision@%d=%.3f  mrr=%.3f",
            flat["count"],
            DEFAULT_K, flat.get(f"recall@{DEFAULT_K}", 0),
            DEFAULT_K, flat.get(f"precision@{DEFAULT_K}", 0),
            flat["mrr"],
        )

    return results


def evaluate(
    model,
    index,
    dataset_path: str | None = None,
    images_base: str = "",
    k_values: list[int] | None = None,
    image_transform=None,
    tokenizer=None,
    k: int = DEFAULT_K,
    verbose: bool = True,
) -> dict[str, ModeMetrics]:
    """Полный пайплайн оценки: модель + ANN-индекс → метрики по трём режимам.

    Args:
        model:            объект, реализующий ``Encoder`` (models/base.py).
        index:            объект с методом ``search(vec, k) → [(image_id, score)]``.
        dataset_path:     путь к val_dataset.csv; None → путь по умолчанию.
        images_base:      префикс для путей изображений, например
                          ``"data/raw/dataset_1M"``.
        k_values:         список K для метрик (по умолчанию [1, 5, 10]).
        image_transform:  callable(PIL.Image) → Tensor(1, C, H, W).
                          Если None — попытается использовать ``model.preprocess_image``.
        tokenizer:        callable(str) → Tensor(1, L).
                          Если None — попытается использовать ``model.tokenize``.
        k:                количество результатов, запрашиваемых у индекса (≥ max(k_values)).
        verbose:          логировать прогресс.

    Returns:
        Словарь ``{mode: ModeMetrics}`` для каждого режима + ключ ``"all"``.

    Example::

        from visual_search.evaluation.evaluate import evaluate
        from visual_search.models.registry import build_model
        from visual_search.index.ann import ANNIndex

        model = build_model(config)
        index = ANNIndex.load("experiments/baseline/index.bin")
        results = evaluate(model, index, images_base="data/raw/dataset_1M")
        print(results["txt"].as_flat_dict())
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # Определяем preprocessors: сначала явные аргументы, потом атрибуты модели
    _image_transform = image_transform or getattr(model, "preprocess_image", None)
    _tokenizer = tokenizer or getattr(model, "tokenize", None)

    if _image_transform is None:
        raise ValueError(
            "image_transform не передан и model.preprocess_image не найден. "
            "Передайте image_transform явно."
        )
    if _tokenizer is None:
        raise ValueError(
            "tokenizer не передан и model.tokenize не найден. "
            "Передайте tokenizer явно."
        )

    search_k = max(k_values + [k])

    def search_fn(query: ValQuery) -> list[int]:
        vec = _encode_query(query, model, _tokenizer, _image_transform)
        hits = index.search(vec, k=search_k)
        return [image_id for image_id, _ in hits]

    return evaluate_with_search_fn(
        search_fn=search_fn,
        dataset_path=dataset_path,
        images_base=images_base,
        k_values=k_values,
        verbose=verbose,
    )


def print_report(results: dict[str, ModeMetrics], k: int = DEFAULT_K) -> None:
    """Вывести красивый текстовый отчёт в stdout."""
    header = f"{'Mode':<12} {'N':>6}  {'R@' + str(k):>8}  {'P@' + str(k):>8}  {'MRR':>8}"
    print(header)
    print("-" * len(header))
    for mode in (*MODES, "all"):
        if mode not in results:
            continue
        m = results[mode]
        flat = m.as_flat_dict()
        print(
            f"{mode:<12} {flat['count']:>6}  "
            f"{flat.get(f'recall@{k}', 0):>8.3f}  "
            f"{flat.get(f'precision@{k}', 0):>8.3f}  "
            f"{flat['mrr']:>8.3f}"
        )
