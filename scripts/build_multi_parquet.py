#!/usr/bin/env python3
"""Собрать train_full_multi.parquet: к каждому товару добавить колонку image_paths
= [титульная] + локально присутствующие доп.ракурсы (из extra_image_map.json).

Запускать ПОСЛЕ докачки доп.картинок (rsync need_extra_images.txt)."""
from __future__ import annotations
import json, os
import pandas as pd

BASE = "data/raw/dataset_1M"
TRAIN = "data/interim/train_full.parquet"
EXTRA_MAP = "data/interim/extra_image_map.json"
OUT = "data/interim/train_full_multi.parquet"

def main() -> None:
    train = pd.read_parquet(TRAIN)
    extra_map = json.load(open(EXTRA_MAP))  # str(item_id) -> [rel paths]

    def build(row):
        paths = [row["title_image_path"]]
        for p in extra_map.get(str(row["item_id"]), []):
            if os.path.exists(os.path.join(BASE, p)):
                paths.append(p)
        return paths

    train["image_paths"] = train.apply(build, axis=1)
    lens = train["image_paths"].map(len)
    print(f"товаров: {len(train)} | avg фото/товар: {lens.mean():.2f} | "
          f">1 фото: {(lens > 1).sum()} | max: {lens.max()}")
    train.to_parquet(OUT, index=False)
    print("сохранено:", OUT)

if __name__ == "__main__":
    main()
