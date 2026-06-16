#!/usr/bin/env python3
"""Индексирование каталога: кодируем ТИТУЛЬНЫЕ картинки товаров выбранной моделью,
строим FAISS-индекс (IndexFlatIP по L2-нормированным векторам) и метаданные.
Артефакты для сервиса: catalog.faiss + catalog_meta.parquet (строка i ↔ вектор i).

Один вектор на товар = полное покрытие каталога продуктов (без дублей-ракурсов).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))                       # eval_full
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eval_full import load_model, _CatalogDS
from visual_search.common.io import INTERIM_DIR, RAW_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2_l16_256")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--train-parquet", default=str(INTERIM_DIR / "train_full.parquet"),
                    help="источник item_id + title_image_path (все товары каталога)")
    ap.add_argument("--images-base", default=str(RAW_DIR / "dataset_1M"))
    ap.add_argument("--out-index", default="artifacts/catalog.faiss")
    ap.add_argument("--out-meta", default="artifacts/catalog_meta.parquet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.train_parquet, columns=["item_id", "title_image_path"])
    if args.limit:
        df = df.head(args.limit)
    # только локально присутствующие титульные
    exists = df["title_image_path"].map(lambda p: os.path.exists(os.path.join(args.images_base, p)))
    df = df[exists].drop_duplicates("item_id").reset_index(drop=True)
    print(f"каталог: {len(df)} товаров для индексации")

    model = load_model(args.model, args.ckpt, torch.device(args.device))
    preprocess, _ = model.get_processor()
    ids = list(range(len(df)))
    paths = df["title_image_path"].tolist()
    dl = DataLoader(_CatalogDS(ids, paths, args.images_base, preprocess),
                    batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    vecs, keep_pos, t0 = [], [], time.time()
    with torch.no_grad():
        for imgs, pos in dl:
            ok = pos != -1
            if ok.sum() == 0:
                continue
            vecs.append(model.encode_image(imgs[ok].to(args.device)).cpu().numpy())
            keep_pos.extend(pos[ok].tolist())
    vectors = np.concatenate(vecs).astype(np.float32)
    faiss.normalize_L2(vectors)
    print(f"закодировано {len(vectors)} за {time.time()-t0:.0f}s, dim={vectors.shape[1]}")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    Path(args.out_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.out_index)
    # метаданные в порядке векторов
    meta = df.iloc[keep_pos][["item_id", "title_image_path"]].rename(columns={"title_image_path": "image_path"})
    meta.reset_index(drop=True).to_parquet(args.out_meta, index=False)
    print(f"сохранено: {args.out_index} ({os.path.getsize(args.out_index)/1e6:.0f} MB) + {args.out_meta}")


if __name__ == "__main__":
    main()
