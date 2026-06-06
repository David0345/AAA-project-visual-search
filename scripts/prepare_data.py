"""CLI: подготовка всех датасетов
(EDA-фильтр -> train.parquet -> leakage removal -> val/test split).
За один запуск создаёт готовые train.parquet, val.csv, test.csv.
Запуск:
  # С конфигом по умолчанию
  uv run python scripts/prepare_data.py

  # С переопределением путей через CLI
  uv run python scripts/prepare_data.py \
    +data.data_dir=/path/to/dataset_1M \
    +data.valid_ids=/path/to/valid_image_ids.csv \
    +data.output=data/processed/train.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from visual_search.common.seed import set_seed
from visual_search.common.logging import get_logger
from visual_search.common.io import PROJECT_ROOT, RAW_DIR, INTERIM_DIR, PROCESSED_DIR
from visual_search.data.prepare.build_train import prepare

log = get_logger(__name__)


@hydra.main(config_path='../configs', config_name='config', version_base='1.3')
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)

    if config.seed.get("fix", True):
        set_seed(config.seed.seed, deterministic=config.seed.get("deterministic_algorithms", False))

    data_config = config.data
    data_dir = Path(data_config.get('data_dir', RAW_DIR / 'dataset_1M'))
    valid_ids = Path(data_config.get('valid_ids', INTERIM_DIR / 'valid_image_ids.csv'))
    val_csv_raw = Path(data_config.get('val_csv_raw', PROJECT_ROOT / 'src/visual_search/evaluation/val_dataset/val_dataset.csv'))
    output_dir = Path(data_config.get('output_dir', PROCESSED_DIR))

    if not data_dir.exists():
        log.error(f'Data directory not found: {data_dir}')
        log.error(f'Expected raw data at {RAW_DIR}. Check your config or download the dataset.')
        sys.exit(1)
    if not valid_ids.exists():
        log.error(f'Valid IDs file not found: {valid_ids}')
        sys.exit(1)
    if not val_csv_raw.exists():
        log.error(f'Raw validation CSV not found: {val_csv_raw}')
        sys.exit(1)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_train = output_dir / 'train_raw.parquet'
    log.info('Сборка train.parquet из сырых данных...')

    try:
        original_argv = sys.argv.copy()

        sys.argv = [
            'build_train.py',
            '--data-dir', str(data_dir),
            '--valid-ids', str(valid_ids),
            '--output', str(tmp_train),
        ]

        prepare()
        sys.argv = original_argv

    except Exception as e:
        log.exception(f'Failed to build train.parquet: {e}')
        sys.exit(1)

    log.info('Проверка item_id-утечек и фильтрация...')
    train_df = pd.read_parquet(tmp_train)
    val_df = pd.read_csv(val_csv_raw)

    train_items = set(train_df['item_id'])
    val_items = set(val_df['item_id'])
    overlap = train_items & val_items
    if overlap:
        train_df = train_df[~train_df['item_id'].isin(overlap)]

    log.info('Сплит валидации на val.csv и test.csv...')
    test_frac = data_config.get('test_frac', 0.3)

    unique_items = val_df['item_id'].dropna().unique()
    np.random.shuffle(unique_items)

    n_test = max(1, int(len(unique_items) * test_frac))
    test_items_set = set(unique_items[:n_test])
    val_items_set = set(unique_items[n_test:])

    final_val = val_df[val_df['item_id'].isin(val_items_set)].reset_index(drop=True)
    final_test = val_df[val_df['item_id'].isin(test_items_set)].reset_index(drop=True)

    log.info(f'   Уникальных товаров: {len(unique_items)}, val: {len(val_items_set)}, test: {len(test_items_set)}')
    log.info(f'   Строк в файлах: val: {len(final_val)}, test: {len(final_test)}')

    final_train_path = output_dir / 'train.parquet'
    final_val_path = output_dir / 'val.csv'
    final_test_path = output_dir / 'test.csv'

    train_df.to_parquet(final_train_path, index=False)
    final_val.to_csv(final_val_path, index=False)
    final_test.to_csv(final_test_path, index=False)

    tmp_train.unlink(missing_ok=True)

    log.info(f'Данные сохранены в {output_dir}:')
    log.info(f'   train.parquet : {len(train_df):,} строк')
    log.info(f'   val.csv       : {len(final_val):,} строк')
    log.info(f'   test.csv      : {len(final_test):,} строк')


if __name__ == "__main__":
    main()
