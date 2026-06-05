"""CLI: сборка ANN-индекса из эмбеддингов каталога.

Тонкая обёртка — логика в visual_search.index.build_index.
Запуск: python scripts/build_index.py --checkpoint experiments/<run_id>/checkpoints/best.pt
"""

from __future__ import annotations

from visual_search.common.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="data/processed/index")
    args = parser.parse_args()
    raise NotImplementedError("TODO(index): вызвать index.build_index")


if __name__ == "__main__":
    main()
