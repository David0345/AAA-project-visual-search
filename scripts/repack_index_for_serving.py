#!/usr/bin/env python3
"""Переупаковать наш индекс (catalog.faiss позиционный + catalog_meta.parquet) в формат
serving-бэкенда: ANNIndex-директория (index.faiss + ids.npy + meta.json, ключ = item_id)
+ metadata.parquet (item_id, image_path, product_text, param2, brand).

Векторы переиспользуются из catalog.faiss (НЕ перекодируем). Метаданные обогащаем из
train_full.parquet. Запуск на сервере приложения, где лежат артефакты.

    python scripts/repack_index_for_serving.py \
        --faiss vs_artifacts/catalog.faiss \
        --catalog-meta vs_artifacts/catalog_meta.parquet \
        --train-full vs_artifacts/interim/train_full.parquet \
        --out-dir vs_artifacts/serving_index
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.index.ann import ANNIndex, IndexSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", default="vs_artifacts/catalog.faiss")
    ap.add_argument("--catalog-meta", default="vs_artifacts/catalog_meta.parquet")
    ap.add_argument("--train-full", default="vs_artifacts/interim/train_full.parquet")
    ap.add_argument("--out-dir", default="vs_artifacts/serving_index")
    args = ap.parse_args()

    # 1) векторы из готового faiss (позиционный IndexFlatIP) — реконструируем
    idx = faiss.read_index(args.faiss)
    n, dim = idx.ntotal, idx.d
    vectors = idx.reconstruct_n(0, n).astype(np.float32)
    faiss.normalize_L2(vectors)              # на всякий случай (индекс уже норм.)
    print(f"векторов: {n}, dim: {dim}")

    # 2) item_id в порядке строк индекса
    cmeta = pd.read_parquet(args.catalog_meta)   # строка i ↔ вектор i: item_id, image_path
    assert len(cmeta) == n, f"meta {len(cmeta)} != index {n}"
    ids = cmeta["item_id"].astype("int64").to_numpy()

    # 3) ANNIndex с ключом item_id
    ann = ANNIndex(embed_dim=dim, spec=IndexSpec(backend="flat")).build(vectors, ids)
    ann.save(args.out_dir)

    # 4) metadata.parquet: image_path + (опц.) product_text/param2/brand из train_full
    md = cmeta.copy()
    if Path(args.train_full).exists():
        tf = pd.read_parquet(args.train_full, columns=["item_id", "product_text", "param2", "brand"])
        md = md.merge(tf, on="item_id", how="left")
    else:
        print(f"WARN: {args.train_full} нет — метаданные только item_id+image_path")
        for c in ("product_text", "param2", "brand"):
            md[c] = None
    md["item_id"] = md["item_id"].astype("int64")
    md = md[["item_id", "image_path", "product_text", "param2", "brand"]]
    md.to_parquet(Path(args.out_dir) / "metadata.parquet", index=False)
    print(f"metadata: {len(md)} строк, колонки {list(md.columns)} -> {args.out_dir}/metadata.parquet")
    print("готово:", args.out_dir)


if __name__ == "__main__":
    main()
