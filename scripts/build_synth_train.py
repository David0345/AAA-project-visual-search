#!/usr/bin/env python3
"""Собрать train-parquet с LLM-запросами: берём train_full, оставляем товары,
для которых есть queries_llm (от Qwen2-VL), и подменяем колонку queries на них.
Колонка title_image_path/прочее сохраняются → finetune_mini работает как есть."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import INTERIM_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default=str(INTERIM_DIR / "train_full.parquet"))
    ap.add_argument("--llm-queries", default=str(INTERIM_DIR / "queries_qwen.parquet"))
    ap.add_argument("--out", default=str(INTERIM_DIR / "train_synth.parquet"))
    ap.add_argument("--min-queries", type=int, default=2, help="мин. число LLM-запросов на товар")
    args = ap.parse_args()

    train = pd.read_parquet(args.train_parquet)
    llm = pd.read_parquet(args.llm_queries)
    # parquet возвращает списки как np.ndarray → проверяем по len(), а не isinstance
    llm = llm[llm["queries_llm"].map(lambda x: x is not None and len(x) >= args.min_queries)]
    print(f"train_full: {len(train)} | llm-запросы (>= {args.min_queries}): {len(llm)}")

    merged = train.merge(llm, on="item_id", how="inner")
    merged["queries"] = merged["queries_llm"]
    merged = merged.drop(columns=["queries_llm"])
    print(f"итог train_synth: {len(merged)} товаров | avg queries: "
          f"{merged['queries'].map(len).mean():.1f}")
    merged.to_parquet(args.out, index=False)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
