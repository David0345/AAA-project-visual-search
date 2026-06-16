"""Zero-shot оценка претренированного CLIP на val_dataset без обучения.

Запускать ПЕРЕД первым обучением — устанавливает нижнюю/верхнюю планку.
Результат сохраняется в experiments/zeroshot/<model>_<pretrained>/metrics.json.

Примеры:
    # EN CLIP (стандарт):
    uv run python scripts/zeroshot_eval.py

    # Multilingual CLIP:
    uv run python scripts/zeroshot_eval.py --model xlm_clip_vit_b32

    # С маленьким каталогом для быстрой проверки:
    uv run python scripts/zeroshot_eval.py --max-catalog 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.seed import set_seed
from visual_search.common.io import PROJECT_ROOT, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="clip_vit_b32",
                        choices=["clip_vit_b32", "clip_vit_b16", "xlm_clip_vit_b32"],
                        help="Имя модели из registry")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-csv", default=PROJECT_ROOT / "src/visual_search/evaluation/val_dataset/val_dataset.csv",
                        help="Путь к val_dataset.csv")
    parser.add_argument("--images-base", default=RAW_DIR / "dataset_1M",
                        help="Корень каталога изображений")
    parser.add_argument("--max-catalog", type=int, default=None,
                        help="Ограничить каталог N случайными товарами (для быстрой проверки)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--out-dir", default="experiments/zeroshot")
    args = parser.parse_args()

    set_seed(42, deterministic=False)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    # Модель
    from visual_search.models.registry import build_model
    from visual_search.models import encoders  # noqa: F401 — регистрация

    model_config = {"name": args.model}
    log.info("Загружаем модель: %s ...", args.model)
    model = build_model(model_config).to(device)
    model.eval()
    log.info("embed_dim=%d", model.embed_dim)

    # Val dataset
    from visual_search.evaluation.val_dataset import ValDataset

    ds_kwargs = {"csv_path": args.val_csv} if args.val_csv else {}
    dataset = ValDataset(**ds_kwargs, images_base=args.images_base)
    log.info("Val dataset: %s", dataset.stats())

    # ANN индекс: строим маленький flat-индекс из val-каталога
    # (эмбеддинги уникальных image_id из таргетов + distractor'ы)
    from visual_search.index.ann import ANNIndex, IndexSpec
    from PIL import Image
    import os

    log.info("Собираем каталог из val-таргетов ...")
    target_image_ids: set[int] = set()
    image_id_to_path: dict[int, str] = {}

    for i in range(len(dataset)):
        q = dataset[i]
        target_image_ids.update(q.target_image_ids)
        # image_id запроса тоже добавляем в каталог
        if q.image_id is not None and q.image_path is not None:
            image_id_to_path[q.image_id] = q.image_path

    # Для image-запросов таргеты = другие изображения. Нужно подтянуть пути.
    # Пути для таргетов берём из val CSV напрямую:
    import pandas as pd
    csv_path = args.val_csv or str(
        Path(__file__).parent.parent
        / "src/visual_search/evaluation/val_dataset/val_dataset.csv"
    )
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        iid = int(row["image_id"]) if pd.notna(row.get("image_id")) else None
        ipath = str(row["image_path"]) if pd.notna(row.get("image_path")) else None
        if iid is not None and ipath is not None:
            image_id_to_path[iid] = ipath

    # Собираем уникальные image_id для каталога
    catalog_ids = sorted(image_id_to_path.keys())
    if args.max_catalog:
        import random
        random.shuffle(catalog_ids)
        catalog_ids = catalog_ids[: args.max_catalog]
        # Убеждаемся, что таргеты присутствуют
        for q_idx in range(len(dataset)):
            for tid in dataset[q_idx].target_image_ids:
                if tid in image_id_to_path and tid not in catalog_ids:
                    catalog_ids.append(tid)

    log.info("Каталог: %d изображений", len(catalog_ids))

    # Кодируем каталог
    log.info("Кодируем каталог ...")
    catalog_vecs: list[np.ndarray] = []
    valid_ids: list[int] = []

    batch_size = 64
    for start in range(0, len(catalog_ids), batch_size):
        batch_ids = catalog_ids[start: start + batch_size]
        tensors: list[torch.Tensor] = []
        ok_ids: list[int] = []
        for iid in batch_ids:
            img_path = os.path.join(args.images_base, image_id_to_path[iid])
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = model.preprocess_image(img).squeeze(0)
                tensors.append(tensor)
                ok_ids.append(iid)
            except Exception as e:
                log.debug("Пропускаем %s: %s", img_path, e)

        if not tensors:
            continue
        with torch.no_grad():
            emb = model.encode_image(torch.stack(tensors).to(device))
        catalog_vecs.append(emb.cpu().numpy())
        valid_ids.extend(ok_ids)

        if (start // batch_size) % 10 == 0:
            log.info("  %d / %d", start + len(batch_ids), len(catalog_ids))

    vectors = np.concatenate(catalog_vecs, axis=0).astype(np.float32)
    ids_arr = np.array(valid_ids, dtype=np.int64)

    # Строим flat-индекс (точный)
    index = ANNIndex(embed_dim=model.embed_dim, spec=IndexSpec(backend="flat"))
    index.build(vectors, ids_arr)
    log.info("Индекс: %d векторов", index.ntotal)

    # Функция поиска для evaluate_with_search_fn
    def search_fn(query):
        """Кодируем запрос и ищем в индексе."""
        mode = query.mode

        vecs: list[np.ndarray] = []
        weights: list[float] = []

        if query.image_path is not None and mode in ("image", "multimodal"):
            try:
                img = Image.open(query.image_path).convert("RGB")
                t = model.preprocess_image(img).to(device)
                with torch.no_grad():
                    v = model.encode_image(t).squeeze(0).cpu().numpy()
                vecs.append(v)
                weights.append(0.5 if mode == "multimodal" else 1.0)
            except Exception as e:
                log.debug("image encode error: %s", e)

        if query.txt_query is not None and mode in ("txt", "multimodal"):
            tokens = model.tokenize(query.txt_query).to(device)
            with torch.no_grad():
                v = model.encode_text(tokens).squeeze(0).cpu().numpy()
            vecs.append(v)
            weights.append(0.5 if mode == "multimodal" else 1.0)

        if not vecs:
            return []

        vec = np.average(vecs, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm

        hits = index.search(vec, k=max(args.k_values))
        return [iid for iid, _ in hits]

    # Оцениваем — вручную по режимам (не зависим от SearchEvalDataset)
    log.info("Запускаем оценку ...")
    from visual_search.evaluation.metrics import aggregate, ModeMetrics
    MODES = ("image", "txt", "multimodal")
    results: dict[str, ModeMetrics] = {}
    all_ranks, all_targets, all_cats = [], [], []

    for mode in MODES:
        queries = dataset.get_by_mode(mode)
        if not queries:
            continue
        ranks, targets, cats = [], [], []
        for q in queries:
            ranked = search_fn(q)
            ranks.append(ranked)
            targets.append(q.target_image_ids)
            cats.append(str(q.metadata.get("param2") or "unknown"))
        results[mode] = aggregate(ranks, targets, args.k_values, cats, mode=mode)
        m = results[mode]
        log.info("[%s] n=%d  recall@10=%.3f  precision@10=%.3f  mrr=%.3f",
                 mode, m.count,
                 m.recall_at_k.get(10, 0), m.precision_at_k.get(10, 0), m.mrr_score)
        all_ranks += ranks; all_targets += targets; all_cats += cats

    results["all"] = aggregate(all_ranks, all_targets, args.k_values, all_cats, mode="all")

    # Выводим таблицу
    from visual_search.evaluation.evaluate import print_report
    print()
    print(f"Zero-shot: {args.model}")
    print_report(results)

    # Сохраняем
    out_dir = Path(args.out_dir) / f"{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dict = {mode: m.as_flat_dict() for mode, m in results.items()}
    out_file = out_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    log.info("Результаты: %s", out_file)


if __name__ == "__main__":
    main()
