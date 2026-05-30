"""CLI: офлайн-подготовка данных (EDA-фильтр + сборка train.parquet).

Тонкая обёртка — вся логика в visual_search.data.prepare.
Запуск: python scripts/prepare_data.py --config configs/data/baseline.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("TODO(Подготовка данных): вызвать data.prepare")


if __name__ == "__main__":
    main()
