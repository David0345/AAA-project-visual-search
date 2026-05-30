"""Пути проекта и чтение/запись артефактов (parquet, csv, эмбеддинги, индексы).

Единая точка правды о том, ГДЕ что лежит (data/raw, data/processed,
experiments/<run_id>/...), чтобы пути не были раскиданы строками по коду.

TODO(common): PROJECT_ROOT, helpers read_parquet/save_embeddings/run_dir.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
