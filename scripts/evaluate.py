"""CLI: оценка модели на val_dataset.

Тонкая обёртка — логика в visual_search.evaluation.evaluate.
Запуск: python scripts/evaluate.py --checkpoint ... --index data/processed/index
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    raise NotImplementedError("TODO(Оценка): загрузить модель+индекс, вызвать evaluate")


if __name__ == "__main__":
    main()
