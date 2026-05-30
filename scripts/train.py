"""CLI: запуск обучения.

Тонкая обёртка — логика в visual_search.training.train.run.
Запуск: python scripts/train.py --config configs/experiment/clip_baseline.yaml
"""

from __future__ import annotations

import argparse

from visual_search.common.config import load_config
from visual_search.training.train import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
