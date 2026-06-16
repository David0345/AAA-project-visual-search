#!/usr/bin/env python3
"""Из gemini_item_queries.csv делаем:
  1) held-out Gemini diverse-eval (val_dataset-формат, txt-режим) — товары, которые
     НЕ пойдут в обучение (чистая независимая валидация на запросах третьей модели);
  2) gemini_train.parquet (item_id, queries_llm) — остальные товары для микса в обучение.

Запросы Gemini — 5 на товар, разнообразные. Релевантность авто: запрос → титульная
картинка этого товара (target = её image_id)."""
from __future__ import annotations
import argparse, ast, os, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import INTERIM_DIR

IMAGES_CSV = "/home/vasiutinpasha/personal/images.csv"
BASE = "data/raw/dataset_1M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini-csv", default=str(INTERIM_DIR / "gemini_item_queries.csv"))
    ap.add_argument("--eval-items", type=int, default=450, help="сколько товаров в held-out eval")
    ap.add_argument("--eval-out", default=str(INTERIM_DIR / "gemini_eval.csv"))
    ap.add_argument("--train-out", default=str(INTERIM_DIR / "gemini_train_queries.parquet"))
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    g = pd.read_csv(args.gemini_csv)
    g["queries"] = g["queries"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # титульная картинка товара: image_id + path
    imgs = pd.read_csv(IMAGES_CSV, usecols=["item_id", "image_id", "is_title", "image_path"])
    titles = imgs[imgs.is_title == True].drop_duplicates("item_id")[["item_id", "image_id", "image_path"]]
    g = g.merge(titles, on="item_id", how="inner")
    g = g[g.image_path.map(lambda p: os.path.exists(os.path.join(BASE, p)))].reset_index(drop=True)
    print(f"пригодных Gemini-товаров (с локальной титульной): {len(g)}")

    g = g.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_df = g.iloc[:args.eval_items]
    train_df = g.iloc[args.eval_items:]

    # 1) held-out eval в формате val_dataset (txt-режим)
    rows, qid = [], 0
    for r in eval_df.itertuples():
        for q in r.queries:
            rows.append({"query_id": qid, "mode": "txt", "item_id": r.item_id,
                         "image_id": "", "image_path": "", "txt_query": q,
                         "target_images_id": "{%d}" % int(r.image_id),
                         "param2": "", "category_name": "", "sostoyanie": "",
                         "cvet": "", "brand": ""})
            qid += 1
    pd.DataFrame(rows).to_csv(args.eval_out, index=False)
    print(f"held-out eval: {len(eval_df)} товаров, {len(rows)} запросов -> {args.eval_out}")

    # 2) train-запросы для микса
    train_df[["item_id"]].assign(queries_llm=train_df["queries"].values).to_parquet(args.train_out, index=False)
    print(f"gemini train: {len(train_df)} товаров -> {args.train_out}")


if __name__ == "__main__":
    main()
