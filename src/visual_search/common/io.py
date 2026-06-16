"""Пути проекта.
Единая точка правды о том, ГДЕ что лежит (data/raw, data/processed,
experiments/<run_id>/...), чтобы пути не были раскиданы строками по коду.

Ожидаемая структура директорий:
    data/
    ├── raw/          # dataset_1M/ (сырые изображения, items.csv, images.csv)
    ├── interim/      # valid_image_ids.csv, tmp_файлы EDA
    └── processed/    # train.parquet, val.csv, test.csv (готовые датасеты)

    experiments/
    └── <run_id>/     # checkpoints/, metrics.json, config.yaml, train.log
"""

from __future__ import annotations
from pathlib import Path


def _get_project_root() -> Path:
    """Находит корень проекта по наличию pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _get_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

CONFIGS_DIR = PROJECT_ROOT / "configs"
