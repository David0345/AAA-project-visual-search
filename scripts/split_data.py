#!/usr/bin/env python3
"""
Генерирует val.csv и test.csv из исходного val_dataset.csv и train.parquet
без пересечений по item_id.
Сохраняет распределение по mode.

Пример запуска:
    python scripts/split_data.py \
        --train-parquet data/interim/train.parquet \
        --val-csv src/visual_search/evaluation/val_dataset/val_dataset.csv \
        --output-dir data/processed
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_splits(train_path: Path, val_path: Path, out_dir: Path,
                  test_frac: float = 0.3, seed: int = 42):
    logging.info('Loading datasets...')
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_csv(val_path)

    train_items = set(train_df['item_id'])
    val_items = set(val_df['item_id'])
    overlap = train_items.intersection(val_items)
    if overlap:
        final_train = train_df[~train_df['item_id'].isin(overlap)]

    final_train.to_parquet(out_dir / 'final_train.parquet', index=False)

    # стратифицированный split по mode
    np.random.seed(seed)
    val_rows, test_rows = [], []
    for mode in val_df['mode'].unique():
        mode_df = val_df[val_df['mode'] == mode].reset_index(drop=True)
        n_test = max(1, int(len(mode_df) * test_frac))
        idx = np.random.permutation(len(mode_df))
        test_rows.append(mode_df.iloc[idx[:n_test]])
        val_rows.append(mode_df.iloc[idx[n_test:]])

    final_val = pd.concat(val_rows, ignore_index=True)
    final_test = pd.concat(test_rows, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    final_val.to_csv(out_dir / 'val.csv', index=False)
    final_test.to_csv(out_dir / 'test.csv', index=False)

    logging.info(f'Splits created. Train: {len(final_train)}, Val: {len(final_val)}, Test: {len(final_test)}')
    logging.info(f'Сохранено в: {out_dir}')
    return final_train, final_val, final_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-parquet', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--test-frac', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    create_splits(args.train_parquet, args.val_csv, args.output_dir, args.test_frac, args.seed)
