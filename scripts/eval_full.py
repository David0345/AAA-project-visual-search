#!/usr/bin/env python3
"""Реалистичный retrieval-eval по БОЛЬШОМУ каталогу (без ручной разметки).

Отличие от run_eval (477 картинок): каталог = val-запросы + val-таргеты
(пути берём из полного images.csv) + N дистракторов из локального пула.
Так оживает image-режим (таргеты image-запроса = другие фото того же товара
теперь реально лежат в каталоге) и метрики становятся честнее (поиск среди
десятков тысяч, а не сотен).

Ground truth — авто: «релевантно = тот же item» (target_images_id в
val_dataset.csv = другие image_id того же товара). Ручной разметки нет.

Примеры:
    # baseline (zero-shot)
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_full.py \
        --model xlm_clip_vit_b32 --catalog-size 50000 --device cuda \
        --out-name baseline_50k
    # дообученная модель
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_full.py \
        --model xlm_clip_vit_b32 \
        --ckpt experiments/fullsweep_lr5e-6/best_ep3_model_only.pt \
        --catalog-size 50000 --device cuda --out-name winner_50k
"""
from __future__ import annotations

import argparse, json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.seed import set_seed
from visual_search.common.io import PROJECT_ROOT, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

VAL_CSV = PROJECT_ROOT / "src/visual_search/evaluation/val_dataset/val_dataset.csv"
# CSV всего каталога (image_id, image_path); переопределяется флагом --images-csv
DEFAULT_IMAGES_CSV = RAW_DIR / "dataset_1M" / "images.csv"
VALID_IDS = PROJECT_ROOT / "src/visual_search/data/eda/valid_image_ids.csv"


def parse_targets(s: str) -> set[int]:
    s = str(s).strip()
    if s.startswith("{") and s.endswith("}"):
        return {int(x) for x in s[1:-1].split(",") if x.strip()}
    return set()


def build_catalog(images_base: str, catalog_size: int, seed: int, val_csv=VAL_CSV,
                  images_csv=DEFAULT_IMAGES_CSV, valid_ids_csv=VALID_IDS) -> dict[int, str]:
    """image_id -> относительный путь. Гарантированно включает val-запросы+таргеты,
    добивает дистракторами из локально присутствующих EDA-валидных картинок."""
    val = pd.read_csv(val_csv)
    need: set[int] = set(int(x) for x in val["image_id"].dropna())
    for s in val["target_images_id"].dropna():
        need |= parse_targets(s)

    valid = set(pd.read_csv(valid_ids_csv)["image_id"])
    rng = np.random.default_rng(seed)

    id2path: dict[int, str] = {}
    distractor_pool: list[tuple[int, str]] = []
    for ch in pd.read_csv(images_csv, usecols=["image_id", "image_path"], chunksize=200_000):
        # нужные (val) — берём всегда, проверим существование ниже
        sub_need = ch[ch.image_id.isin(need)]
        for iid, p in zip(sub_need.image_id, sub_need.image_path):
            id2path[int(iid)] = p
        # пул дистракторов: EDA-валидные, не из val
        sub = ch[ch.image_id.isin(valid) & ~ch.image_id.isin(need)]
        for iid, p in zip(sub.image_id, sub.image_path):
            distractor_pool.append((int(iid), p))

    # оставляем только локально присутствующие
    def local(p): return os.path.exists(os.path.join(images_base, p))

    catalog = {iid: p for iid, p in id2path.items() if local(p)}
    n_val_local = len(catalog)
    log.info("val (запросы+таргеты) локально: %d / %d", n_val_local, len(need))

    rng.shuffle(distractor_pool)
    added = 0
    for iid, p in distractor_pool:
        if added >= catalog_size:
            break
        if local(p):
            catalog[iid] = p
            added += 1
    log.info("каталог: %d (val %d + дистракторов %d)", len(catalog), n_val_local, added)
    return catalog


class _CatalogDS(Dataset):
    def __init__(self, ids, paths, base, preprocess):
        self.ids, self.paths, self.base, self.pre = ids, paths, base, preprocess

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        try:
            img = Image.open(os.path.join(self.base, self.paths[i])).convert("RGB")
            return self.pre(img), self.ids[i]
        except Exception:
            return torch.zeros(3, 224, 224), -1


def load_model(model_name, ckpt, device):
    from visual_search.models.registry import build_model
    from visual_search.models import encoders  # noqa: F401
    model = build_model({"name": model_name}).to(device)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        state = state.get("model_state", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("ckpt загружен: %s (missing=%d unexpected=%d)", ckpt, len(missing), len(unexpected))
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xlm_clip_vit_b32")
    ap.add_argument("--ckpt", default=None, help="путь к *_model_only.pt (если дообученная)")
    ap.add_argument("--catalog-size", type=int, default=50000, help="число дистракторов")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--val-csv", default=str(VAL_CSV), help="кастомный val CSV (напр. Gemini held-out)")
    ap.add_argument("--mm-image-weight", type=float, default=0.25, help="вес картинки в multimodal-склейке (0.25 — оптимум по свипу)")
    ap.add_argument("--mm-sweep", action="store_true", help="свип веса склейки multimodal и выход")
    ap.add_argument("--txt-override", default=None, help="JSON {query_id: text} — замена текста запроса (напр. перевод RU→EN)")
    ap.add_argument("--images-base", default=str(RAW_DIR / "dataset_1M"))
    ap.add_argument("--images-csv", default=str(DEFAULT_IMAGES_CSV), help="CSV каталога (image_id, image_path)")
    ap.add_argument("--valid-ids", default=str(VALID_IDS), help="CSV валидных image_id (пул дистракторов)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device(args.device)

    model = load_model(args.model, args.ckpt, device)
    preprocess, _ = model.get_processor()

    # 1) каталог
    catalog = build_catalog(args.images_base, args.catalog_size, args.seed, val_csv=args.val_csv,
                            images_csv=args.images_csv, valid_ids_csv=args.valid_ids)
    ids = list(catalog.keys()); paths = [catalog[i] for i in ids]

    # 2) кодируем каталог через DataLoader
    log.info("кодируем каталог (%d) ...", len(ids))
    t0 = time.time()
    ds = _CatalogDS(ids, paths, args.images_base, preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    vecs, vids = [], []
    with torch.no_grad():
        for imgs, batch_ids in dl:
            keep = batch_ids != -1
            if keep.sum() == 0:
                continue
            emb = model.encode_image(imgs[keep].to(device))
            vecs.append(emb.cpu().numpy())
            vids.extend(batch_ids[keep].tolist())
    vectors = np.concatenate(vecs).astype(np.float32)
    log.info("каталог закодирован: %d векторов за %.0fs", len(vids), time.time() - t0)

    from visual_search.index.ann import ANNIndex, IndexSpec
    index = ANNIndex(embed_dim=model.embed_dim, spec=IndexSpec(backend="flat"))
    index.build(vectors, np.array(vids, dtype=np.int64))

    # 3) eval по режимам
    from visual_search.evaluation.val_dataset import ValDataset
    from visual_search.evaluation.metrics import aggregate, ModeMetrics
    dataset = ValDataset(csv_path=args.val_csv, images_base=args.images_base)

    txt_override = json.load(open(args.txt_override)) if args.txt_override else {}
    if txt_override:
        log.info("txt-override: %d замен текста (напр. перевод)", len(txt_override))

    def query_text(q):
        return txt_override.get(str(q.query_id), q.txt_query)

    def search_fn(q, mm_w_img=None):
        """mm_w_img — вес картинки в multimodal-склейке (текст = 1-w). None → args."""
        w_img = args.mm_image_weight if mm_w_img is None else mm_w_img
        qtext = query_text(q)
        v_list, w = [], []
        if q.image_path is not None and q.mode in ("image", "multimodal"):
            try:
                t = model.preprocess_image(Image.open(q.image_path).convert("RGB")).to(device)
                with torch.no_grad():
                    v_list.append(model.encode_image(t).squeeze(0).cpu().numpy())
                w.append(w_img if q.mode == "multimodal" else 1.0)
            except Exception:
                pass
        if qtext is not None and q.mode in ("txt", "multimodal"):
            with torch.no_grad():
                v_list.append(model.encode_text(model.tokenize(qtext).to(device)).squeeze(0).cpu().numpy())
            w.append((1.0 - w_img) if q.mode == "multimodal" else 1.0)
        if not v_list or sum(w) <= 0:
            return []
        vec = np.average(v_list, axis=0, weights=w).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 1e-8:
            vec /= n
        return [iid for iid, _ in index.search(vec, k=max(args.k_values))]

    # опциональный свип веса склейки multimodal (реюзает уже построенный индекс)
    if args.mm_sweep:
        mm_qs = dataset.get_by_mode("multimodal")
        tgts = [q.target_image_ids for q in mm_qs]
        cts = [str(q.metadata.get("param2") or "unknown") for q in mm_qs]
        log.info("=== mm fusion weight sweep (w_img) ===")
        for wi in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
            ranks = [search_fn(q, mm_w_img=wi) for q in mm_qs]
            mm = aggregate(ranks, tgts, args.k_values, cts, mode="multimodal")
            log.info("  w_img=%.2f  mm MRR=%.3f  R@10=%.3f", wi, mm.mrr_score, mm.recall_at_k.get(10, 0))
        return

    results: dict[str, ModeMetrics] = {}
    all_r, all_t, all_c = [], [], []
    for mode in ("image", "txt", "multimodal"):
        qs = dataset.get_by_mode(mode)
        if not qs:
            continue
        ranks, targets, cats = [], [], []
        for q in qs:
            ranks.append(search_fn(q))
            targets.append(q.target_image_ids)
            cats.append(str(q.metadata.get("param2") or "unknown"))
        results[mode] = aggregate(ranks, targets, args.k_values, cats, mode=mode)
        m = results[mode]
        log.info("[%s] n=%d R@10=%.3f P@10=%.3f MRR=%.3f", mode, m.count,
                 m.recall_at_k.get(10, 0), m.precision_at_k.get(10, 0), m.mrr_score)
        all_r += ranks; all_t += targets; all_c += cats
    results["all"] = aggregate(all_r, all_t, args.k_values, all_c, mode="all")
    m = results["all"]
    log.info("[all] R@10=%.3f MRR=%.3f", m.recall_at_k.get(10, 0), m.mrr_score)

    # 4) сохраняем + ledger
    out_dir = Path("experiments/eval_full") / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = {mode: mm.as_flat_dict() for mode, mm in results.items()}
    json.dump(flat, open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)

    ledger = {
        "out_name": args.out_name, "model": args.model, "ckpt": args.ckpt,
        "catalog_total": len(vids), "catalog_distractors": args.catalog_size,
        "metrics": {mode: {"mrr": mm.mrr_score, "r@10": mm.recall_at_k.get(10, 0)}
                    for mode, mm in results.items()},
    }
    with open("experiments/eval_full_ledger.jsonl", "a") as f:
        f.write(json.dumps(ledger, ensure_ascii=False) + "\n")
    log.info("сохранено: %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
