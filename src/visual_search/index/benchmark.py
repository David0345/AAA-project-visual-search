"""Подбор конфигурации индекса по оценке на валидации.

Отвечает на три вопроса проекта эмпирически, прогоняя val_dataset:

  1. Какой ANN-алгоритм:    flat / ivf / hnsw  (скорость vs качество);
  2. Использовать квантование: ivf vs ivfpq    (RAM vs качество);
  3. title или mean pooling: один титульный вектор на товар или среднее по
     всем изображениям товара (качество поиска).

Методика (чтобы не путать факторы):
  * Этап A (pooling) — на ТОЧНОМ flat-индексе, чтобы аппроксимация ANN не
    искажала сравнение качества. Берём лучший pooling по recall@10.
  * Этап B (алгоритм+квантование) — фиксируем лучший pooling, строим
    flat (эталон) и приближённые индексы; меряем recall@10, recall-vs-exact,
    latency (на запрос), размер в RAM. Рекомендуем самый дешёвый вариант,
    чей recall в пределах допуска от эталона.

Каталог индексируется на уровне товара; таргеты val заданы в image_id, поэтому
мост image_id -> item_id строится из images.csv. Метрики считаются на уровне
товара (товар релевантен, если ему принадлежит хотя бы одна таргетная картинка).
"""

from __future__ import annotations

import ast
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from visual_search.common.logging import get_logger
from visual_search.index.ann import ANNIndex, IndexSpec
from visual_search.index.build_index import (
    load_images_csv,
    load_model_from_checkpoint,
    resolve_device,
)
from visual_search.index.embed import embed_catalog, embed_images

log = get_logger(__name__)

# Какие backend'ы перебираем на этапе B. Пары ivf/ivfpq дают ответ про квантование.
DEFAULT_BACKENDS = ("flat", "ivf", "ivfpq", "hnsw")
DEFAULT_K_VALUES = (1, 5, 10)
RECALL_TOLERANCE = 0.02  # допустимая просадка recall@10 vs точный flat


# --------------------------------------------------------------------------
# Загрузка валидации
# --------------------------------------------------------------------------
@dataclass
class _Query:
    mode: str
    image_path: str | None
    txt_query: str | None
    target_items: set[int]
    source_item: int
    category: str


def _parse_target_ids(raw) -> set[int]:
    """'{1, 2, 3}' / '1;2;3' -> {1,2,3}. Терпим к обоим форматам."""
    if pd.isna(raw):
        return set()
    s = str(raw).strip()
    try:
        val = ast.literal_eval(s)
        return {int(x) for x in val}
    except (ValueError, SyntaxError):
        return {int(x) for x in s.strip("{}[]").replace(";", ",").split(",") if x.strip()}


def load_val_queries(
    val_csv: str | Path,
    modes: tuple[str, ...],
    image_to_item: dict[int, int],
) -> dict[str, list[_Query]]:
    """Прочитать val_dataset.csv и сгруппировать запросы по режиму.

    Таргетные image_id переводятся в item_id; неизвестные каталогу — отбрасываются.
    """
    df = pd.read_csv(val_csv)
    out: dict[str, list[_Query]] = {m: [] for m in modes}
    skipped = 0
    for _, row in df.iterrows():
        mode = row["mode"]
        if mode not in out:
            continue
        target_imgs = _parse_target_ids(row.get("target_images_id"))
        target_items = {image_to_item[i] for i in target_imgs if i in image_to_item}
        if not target_items:
            skipped += 1
            continue
        out[mode].append(
            _Query(
                mode=mode,
                image_path=row["image_path"] if pd.notna(row.get("image_path")) else None,
                txt_query=row["txt_query"] if pd.notna(row.get("txt_query")) else None,
                target_items=target_items,
                source_item=int(row["item_id"]),
                category=str(row.get("param2", "unknown")),
            )
        )
    for m in modes:
        log.info("val mode=%s: %d запросов", m, len(out[m]))
    if skipped:
        log.warning("Пропущено %d запросов: таргеты вне каталога", skipped)
    return out


# --------------------------------------------------------------------------
# Подвыборка каталога (для прогона на части данных)
# --------------------------------------------------------------------------
def subsample_catalog(
    images_df: pd.DataFrame,
    val_queries: dict[str, list[_Query]],
    sample_items: int | None,
    seed: int = 42,
) -> pd.DataFrame:
    """Оставить sample_items товаров, но ВСЕГДА включить таргетные и исходные.

    Иначе recall на подвыборке был бы искусственно занижен.
    """
    if sample_items is None:
        return images_df

    must_keep: set[int] = set()
    for queries in val_queries.values():
        for q in queries:
            must_keep |= q.target_items
            must_keep.add(q.source_item)

    all_items = images_df["item_id"].unique()
    rng = np.random.default_rng(seed)
    n_fill = max(0, sample_items - len(must_keep))
    pool = np.array([i for i in all_items if i not in must_keep])
    filler = rng.choice(pool, size=min(n_fill, len(pool)), replace=False) if len(pool) else np.array([])
    keep = must_keep | set(int(x) for x in filler)
    sub = images_df[images_df["item_id"].isin(keep)]
    log.info("Подвыборка каталога: %d товаров (из них обязательных %d)", sub["item_id"].nunique(), len(must_keep))
    return sub


# --------------------------------------------------------------------------
# Кодирование запросов
# --------------------------------------------------------------------------
@torch.no_grad()
def encode_queries(
    queries: list[_Query],
    model,
    *,
    images_root: str | Path,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    tokenize: Callable[[list[str]], torch.Tensor] | None,
) -> np.ndarray | None:
    """Вектора запросов (Q, embed_dim) L2-norm. None, если режим невозможен."""
    if not queries:
        return None
    mode = queries[0].mode

    img_vecs = None
    if mode in ("image", "multimodal"):
        paths = [q.image_path for q in queries]
        img_vecs = embed_images(
            model, paths, images_root=images_root, batch_size=batch_size,
            device=device, image_size=image_size, num_workers=num_workers,
        )

    txt_vecs = None
    if mode in ("txt", "multimodal"):
        if tokenize is None:
            log.warning("Режим %s пропущен: не передан tokenize (data/tokenization не готов)", mode)
            return None
        tokens = tokenize([q.txt_query or "" for q in queries]).to(device)
        txt = model.encode_text(tokens).float().cpu().numpy().astype(np.float32)
        txt_vecs = txt

    if mode == "image":
        return img_vecs
    if mode == "txt":
        return txt_vecs
    # multimodal: сумма направлений + ренормировка
    combined = img_vecs + txt_vecs
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return (combined / np.maximum(norms, 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------
# Метрики на уровне товара
# --------------------------------------------------------------------------
def _aggregate_metrics(
    ranked_ids: np.ndarray,
    queries: list[_Query],
    k_values: tuple[int, ...],
    exclude_source: bool = True,
) -> dict:
    """ranked_ids: (Q, K) item_id. -> micro + macro(по param2) recall/precision/mrr."""
    micro = {f"recall@{k}": [] for k in k_values}
    micro.update({f"precision@{k}": [] for k in k_values})
    micro["mrr"] = []
    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for row, q in zip(ranked_ids, queries):
        ranked = [int(i) for i in row if i != -1]
        # источник убираем, только если он сам не является таргетом
        if exclude_source and q.source_item not in q.target_items:
            ranked = [i for i in ranked if i != q.source_item]
        tset = q.target_items

        rr = 0.0
        for rank, iid in enumerate(ranked):
            if iid in tset:
                rr = 1.0 / (rank + 1)
                break
        micro["mrr"].append(rr)
        by_cat[q.category]["mrr"].append(rr)

        for k in k_values:
            hits = len(set(ranked[:k]) & tset)
            rec = hits / len(tset)
            prec = hits / k
            micro[f"recall@{k}"].append(rec)
            micro[f"precision@{k}"].append(prec)
            by_cat[q.category][f"recall@{k}"].append(rec)
            by_cat[q.category][f"precision@{k}"].append(prec)

    micro_avg = {m: float(np.mean(v)) if v else 0.0 for m, v in micro.items()}
    # macro = среднее по категориям (усреднённым внутри)
    macro_avg = {}
    for m in micro:
        cat_means = [np.mean(by_cat[c][m]) for c in by_cat if by_cat[c][m]]
        macro_avg[m] = float(np.mean(cat_means)) if cat_means else 0.0

    return {"count": len(queries), "micro": micro_avg, "macro": macro_avg}


def _evaluate_index(
    index: ANNIndex,
    query_vecs_by_mode: dict[str, np.ndarray | None],
    val_queries: dict[str, list[_Query]],
    k_values: tuple[int, ...],
) -> dict:
    """Метрики индекса по каждому режиму (qvecs уже посчитаны)."""
    k_search = max(k_values) + 10
    report = {}
    for mode, qvecs in query_vecs_by_mode.items():
        if qvecs is None or not val_queries[mode]:
            continue
        ranked, _ = index.batch_search(qvecs, k_search)
        report[mode] = _aggregate_metrics(ranked, val_queries[mode], k_values)
    return report


def _measure_latency(index: ANNIndex, qvecs: np.ndarray, k: int, repeats: int = 1) -> dict:
    """Per-query latency (онлайн-сценарий 1 rps): mean / p95 в мс."""
    # warmup
    index.search(qvecs[0], k)
    times = []
    for _ in range(repeats):
        for i in range(len(qvecs)):
            t0 = time.perf_counter()
            index.search(qvecs[i], k)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95))}


# --------------------------------------------------------------------------
# Главный прогон
# --------------------------------------------------------------------------
def _primary_recall(per_mode_report: dict, k: int = 10) -> float:
    """Сводный recall@k по доступным режимам (micro), среднее."""
    vals = [r["micro"][f"recall@{k}"] for r in per_mode_report.values() if r]
    return float(np.mean(vals)) if vals else 0.0


def run_benchmark(
    checkpoint: str | Path,
    images_csv: str | Path,
    val_csv: str | Path,
    *,
    images_root: str | Path = "data/raw/dataset_1M",
    valid_ids_csv: str | Path | None = None,
    out_dir: str | Path = "data/processed/index_benchmark",
    poolings: tuple[str, ...] = ("title", "mean"),
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    modes: tuple[str, ...] = ("image", "txt", "multimodal"),
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    sample_items: int | None = None,
    device: str = "auto",
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 8,
    tokenize: Callable[[list[str]], torch.Tensor] | None = None,
    seed: int = 42,
) -> dict:
    """Прогнать сетку конфигов и вернуть отчёт с рекомендациями."""
    dev = resolve_device(device)
    model = load_model_from_checkpoint(checkpoint, dev)

    images_df = load_images_csv(images_csv, valid_ids_csv)
    image_to_item = dict(
        zip(images_df["image_id"].astype(int), images_df["item_id"].astype(int))
    )
    val_queries = load_val_queries(val_csv, modes, image_to_item)
    images_df = subsample_catalog(images_df, val_queries, sample_items, seed)

    embed_kw = dict(
        images_root=images_root, device=dev, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )

    # Вектора запросов считаем один раз (от модели, не от индекса).
    qvecs_by_mode = {
        m: encode_queries(val_queries[m], model, tokenize=tokenize, **embed_kw)
        for m in modes
    }

    # ----- Этап A: pooling на точном flat -----
    log.info("=== Этап A: выбор pooling (title vs mean) на точном flat ===")
    pooling_report = {}
    catalog_cache = {}
    for pooling in poolings:
        catalog = embed_catalog(model, images_df, pooling=pooling, **embed_kw)
        catalog_cache[pooling] = catalog
        idx = ANNIndex(model.embed_dim, IndexSpec(backend="flat")).build(
            catalog.vectors, catalog.item_ids
        )
        rep = _evaluate_index(idx, qvecs_by_mode, val_queries, k_values)
        pooling_report[pooling] = rep
        log.info("pooling=%s recall@10=%.4f", pooling, _primary_recall(rep))

    best_pooling = max(poolings, key=lambda p: _primary_recall(pooling_report[p]))
    log.info("-> Лучший pooling: %s", best_pooling)

    # ----- Этап B: алгоритм + квантование при лучшем pooling -----
    log.info("=== Этап B: алгоритм и квантование (pooling=%s) ===", best_pooling)
    catalog = catalog_cache[best_pooling]

    # эталон — точный flat; берём ранжирование для recall-vs-exact
    exact = ANNIndex(model.embed_dim, IndexSpec(backend="flat")).build(
        catalog.vectors, catalog.item_ids
    )
    exact_ranked = {
        m: exact.batch_search(q, max(k_values))[0]
        for m, q in qvecs_by_mode.items() if q is not None
    }

    algo_report = {}
    for backend in backends:
        try:
            idx = ANNIndex(model.embed_dim, IndexSpec(backend=backend))
            t0 = time.perf_counter()
            idx.build(catalog.vectors, catalog.item_ids)
            build_s = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001 - PQ/IVF могут не обучиться на малом N
            log.warning("backend=%s пропущен: %s", backend, exc)
            algo_report[backend] = {"error": str(exc)}
            continue

        metrics = _evaluate_index(idx, qvecs_by_mode, val_queries, k_values)

        # latency и recall-vs-exact по первому доступному режиму
        ref_mode = next((m for m in modes if qvecs_by_mode.get(m) is not None), None)
        latency = _measure_latency(idx, qvecs_by_mode[ref_mode], max(k_values)) if ref_mode else {}
        rve = None
        if ref_mode is not None:
            approx = idx.batch_search(qvecs_by_mode[ref_mode], max(k_values))[0]
            ex = exact_ranked[ref_mode]
            overlaps = [len(set(a) & set(e)) / max(len(e), 1) for a, e in zip(approx, ex)]
            rve = float(np.mean(overlaps))

        algo_report[backend] = {
            "build_sec": round(build_s, 2),
            "size_mb": round(idx.size_bytes() / 1e6, 2),
            "recall@10": round(_primary_recall(metrics), 4),
            "recall_vs_exact": round(rve, 4) if rve is not None else None,
            "latency": {k: round(v, 3) for k, v in latency.items()},
            "per_mode": metrics,
        }

    recommendation = _recommend(best_pooling, algo_report, pooling_report, k=10)

    report = {
        "best_pooling": best_pooling,
        "pooling_report": pooling_report,
        "algo_report": algo_report,
        "recommendation": recommendation,
        "config": {
            "checkpoint": str(checkpoint),
            "sample_items": sample_items,
            "num_catalog_items": int(len(catalog.item_ids)),
            "modes_evaluated": [m for m in modes if qvecs_by_mode.get(m) is not None],
            "device": str(dev),
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(report)
    log.info("Отчёт сохранён: %s", out / "report.json")
    return report


def _recommend(best_pooling: str, algo_report: dict, pooling_report: dict, k: int = 10) -> dict:
    """Свести замеры в три решения с обоснованием."""
    flat = algo_report.get("flat", {})
    flat_recall = flat.get("recall@10", 0.0)
    budget = flat_recall * (1 - RECALL_TOLERANCE)

    # кандидаты: приближённые backend'ы без ошибок и в пределах допуска по recall
    candidates = {
        b: r for b, r in algo_report.items()
        if b != "flat" and "error" not in r and r["recall@10"] >= budget
    }
    if candidates:
        # самый быстрый среди прошедших по качеству
        best_algo = min(candidates, key=lambda b: candidates[b]["latency"].get("p95_ms", 1e9))
    else:
        best_algo = "flat"  # никто не уложился в допуск — берём точный

    # квантование: сравниваем ivf и ivfpq, если оба есть
    ivf = algo_report.get("ivf", {})
    ivfpq = algo_report.get("ivfpq", {})
    use_quant = None
    quant_reason = "ivf/ivfpq не оценивались"
    if "error" not in ivf and "error" not in ivfpq and ivf and ivfpq:
        size_gain = ivf.get("size_mb", 0) / max(ivfpq.get("size_mb", 1e-9), 1e-9)
        recall_drop = ivf.get("recall@10", 0) - ivfpq.get("recall@10", 0)
        if recall_drop <= RECALL_TOLERANCE and size_gain >= 2.0:
            use_quant = True
            quant_reason = f"PQ сжимает RAM в {size_gain:.1f}x при просадке recall {recall_drop:.3f}"
        else:
            use_quant = False
            quant_reason = f"PQ даёт просадку recall {recall_drop:.3f} (сжатие {size_gain:.1f}x) — не оправдано"

    return {
        "algorithm": best_algo,
        "use_quantization": use_quant,
        "pooling": best_pooling,
        "reasons": {
            "algorithm": (
                f"recall@10={algo_report.get(best_algo, {}).get('recall@10')} "
                f"(эталон flat={flat_recall}), p95="
                f"{algo_report.get(best_algo, {}).get('latency', {}).get('p95_ms')} мс"
            ),
            "quantization": quant_reason,
            "pooling": (
                f"recall@10: title={_primary_recall(pooling_report.get('title', {})):.4f} "
                f"mean={_primary_recall(pooling_report.get('mean', {})):.4f}"
            ),
        },
    }


def _print_summary(report: dict) -> None:
    rec = report["recommendation"]
    lines = [
        "",
        "=" * 64,
        "РЕКОМЕНДАЦИИ ПО ИНДЕКСУ",
        "=" * 64,
        f"  1) Алгоритм ANN:   {rec['algorithm']}",
        f"  2) Квантование:    {rec['use_quantization']}",
        f"  3) Pooling:        {rec['pooling']}",
        "-" * 64,
        "Обоснование:",
        f"  алгоритм:    {rec['reasons']['algorithm']}",
        f"  квантование: {rec['reasons']['quantization']}",
        f"  pooling:     {rec['reasons']['pooling']}",
        "-" * 64,
        f"{'backend':<8} {'recall@10':>10} {'vs_exact':>9} {'p95_ms':>8} {'size_mb':>9} {'build_s':>8}",
    ]
    for b, r in report["algo_report"].items():
        if "error" in r:
            lines.append(f"{b:<8} {'ERROR: ' + r['error'][:40]}")
            continue
        lines.append(
            f"{b:<8} {r['recall@10']:>10} {str(r['recall_vs_exact']):>9} "
            f"{r['latency'].get('p95_ms', '-'):>8} {r['size_mb']:>9} {r['build_sec']:>8}"
        )
    lines.append("=" * 64)
    log.info("\n".join(lines))
