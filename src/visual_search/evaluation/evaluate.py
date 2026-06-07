"""Прогон SearchEvalDataset через модель + ANN-индекс → отчёт по трём режимам.

Это логика пайплайна; запускается тонкой обёрткой ``scripts/evaluate.py``.

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
from typing import Callable, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from visual_search.data.dataset import SearchEvalDataset
from visual_search.data.collate import eval_collate_fn
from visual_search.evaluation.metrics import ModeMetrics, aggregate
from visual_search.models.base import Encoder

logger = logging.getLogger(__name__)

MODES = ("image", "txt", "multimodal")
DEFAULT_K_VALUES = [1, 5, 10]


def _encode_query_batch(batch: List[Dict[str, Any]], model: Encoder, device: torch.device) -> np.ndarray:
    """
    Кодирует батч запросов (batch_size=1 из-за eval_collate_fn) в единый L2-нормированный вектор.
    Корректно обрабатывает мультимодальное слияние с весом alpha.

    Args:
        batch: Список из одного словаря, возвращаемого SearchEvalDataset.
        model: Экземпляр модели, реализующий протокол Encoder.
        device: Устройство (cpu/cuda) для вычислений.

    Returns:
        np.ndarray: Вектор размерности (D,), L2-нормированный, тип float32.
    """
    item = batch[0]
    query = item['query']
    mode = item['mode']

    vecs = []
    weights = []

    if query.get('pixel_values') is not None and mode in ('image', 'multimodal'):
        img_tensor = query['pixel_values'].unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            img_vec = model.encode_image(img_tensor).squeeze(0).cpu().numpy()
        vecs.append(img_vec)
        # Для image вес картинки = 1.0, для multimodal вес = (1 - alpha)
        alpha = query.get('multimodal_alpha', 0.5)
        weights.append(1.0 if mode == 'image' else (1.0 - alpha))

    if query.get('input_ids') is not None and mode in ('txt', 'multimodal'):
        input_ids = query['input_ids'].unsqueeze(0).to(device)  # [1, L]
        attention_mask = None
        if query.get('attention_mask') is not None:
            attention_mask = query['attention_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            txt_vec = model.encode_text(input_ids, attention_mask=attention_mask).squeeze(0).cpu().numpy()
        vecs.append(txt_vec)
        # Для txt вес текста = 1.0, для multimodal вес = alpha
        alpha = query.get('multimodal_alpha', 0.5)
        weights.append(1.0 if mode == 'txt' else alpha)

    if not vecs:
        raise ValueError(f"Не удалось закодировать запрос. mode={mode}, query keys={query.keys()}")

    vec = np.average(vecs, axis=0, weights=weights).astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm

    return vec


def evaluate_with_search_fn(
    search_fn: Callable[[Dict[str, Any]], List[int]],
    dataset: SearchEvalDataset,
    k_values: List[int] | None = None,
    verbose: bool = True,
) -> Dict[str, ModeMetrics]:
    """
    Оценить качество поиска, используя произвольную функцию поиска (низкоуровневая точка входа).
    Полезна для тестирования альтернативных стратегий ранжирования без переобучения модели.

    Args:
        search_fn: Функция, принимающая item-словарь из SearchEvalDataset и возвращающая 
                   список image_id в порядке убывания релевантности.
        dataset: Экземпляр SearchEvalDataset.
        k_values: Список K для метрик Recall@K и Precision@K (по умолчанию [1, 5, 10]).
        verbose: Если True, логировать прогресс и итоги в stdout.

    Returns:
        Dict[str, ModeMetrics]: Словарь метрик по каждому режиму ('image', 'txt', 'multimodal') 
                                и общий режим 'all'.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

    results: Dict[str, ModeMetrics] = {}
    all_ranks: List[List[int]] = []
    all_targets: List[set[int]] = []
    all_categories: List[str] = []

    mode_ranks: Dict[str, List[List[int]]] = {m: [] for m in MODES}
    mode_targets: Dict[str, List[set[int]]] = {m: [] for m in MODES}
    mode_cats: Dict[str, List[str]] = {m: [] for m in MODES}

    search_k = max(k_values)

    if verbose:
        logger.info("Начало оценки через search_fn...")

    for batch in dataloader:
        item = batch[0]
        mode = item['mode']
        target_ids = set(item['target_ids'])
        category = item['metadata'].get('category_name') or 'unknown'

        if not target_ids:
            continue

        ranked_ids = search_fn(item)

        ranked_ids = ranked_ids[:search_k]

        mode_ranks[mode].append(ranked_ids)
        mode_targets[mode].append(target_ids)
        mode_cats[mode].append(category)

        all_ranks.append(ranked_ids)
        all_targets.append(target_ids)
        all_categories.append(category)

    for mode in MODES:
        if mode_ranks[mode]:
            results[mode] = aggregate(
                per_query_ranks=mode_ranks[mode],
                per_query_targets=mode_targets[mode],
                k_values=k_values,
                per_query_categories=mode_cats[mode],
                mode=mode
            )
            if verbose:
                m = results[mode]
                logger.info(f"[{mode}] N={m.count}, Recall@10={m.recall_at_k.get(10, 0):.4f}, MRR={m.mrr_score:.4f}")

    if all_ranks:
        results["all"] = aggregate(
            per_query_ranks=all_ranks,
            per_query_targets=all_targets,
            k_values=k_values,
            per_query_categories=all_categories,
            mode="all"
        )
        if verbose:
            m = results["all"]
            logger.info(f"[all] N={m.count}, Recall@10={m.recall_at_k.get(10, 0):.4f}, MRR={m.mrr_score:.4f}")

    return results


def evaluate(
    model: Encoder,
    index: Any,  # Объект с методом search(vec, k)
    dataset: SearchEvalDataset,
    device: torch.device,
    k_values: List[int] | None = None,
    verbose: bool = True,
) -> Dict[str, ModeMetrics]:
    """
    Полный пайплайн оценки: модель + ANN-индекс → метрики по трём режимам.

    Args:
        model: Экземпляр модели, реализующий протокол Encoder.
        index: Объект ANN-индекса с методом search(query_vec, k).
        dataset: Экземпляр SearchEvalDataset.
        device: Устройство для инференса модели.
        k_values: Список K для метрик (по умолчанию [1, 5, 10]).
        verbose: Если True, логировать прогресс.

    Returns:
        Dict[str, ModeMetrics]: Словарь агрегированных метрик.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    search_k = max(k_values)

    def search_fn(item: Dict[str, Any]) -> List[int]:
        """Внутренняя обёртка: кодирует запрос и ищет в индексе."""
        query_vec = _encode_query_batch([item], model, device)
        hits = index.search(query_vec, k=search_k)
        return [int(image_id) for image_id, score in hits]

    return evaluate_with_search_fn(
        search_fn=search_fn,
        dataset=dataset,
        k_values=k_values,
        verbose=verbose,
    )


def print_report(results: Dict[str, ModeMetrics], k: int = 10) -> None:
    """
    Вывести красивый текстовый отчёт в stdout.

    Args:
        results: Словарь метрик, возвращаемый функциями evaluate или evaluate_with_search_fn.
        k: Значение K для отображения в заголовке таблицы (по умолчанию 10).
    """
    header = f"{'Mode':<12} {'N':>6}  {'R@' + str(k):>8}  {'P@' + str(k):>8}  {'MRR':>8}"
    print("\n" + header)
    print("-" * len(header))
    for mode in (*MODES, "all"):
        if mode not in results:
            continue
        m = results[mode]
        flat = m.as_flat_dict()
        print(
            f"{mode:<12} {flat['count']:>6}   "
            f"{flat.get(f'recall@{k}', 0):>8.3f}   "
            f"{flat.get(f'precision@{k}', 0):>8.3f}   "
            f"{flat['mrr']:>8.3f}"
        )
    print("-" * len(header) + "\n")
