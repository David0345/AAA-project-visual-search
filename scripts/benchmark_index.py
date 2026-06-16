"""CLI: подбор конфигурации индекса по оценке на валидации.

Тонкая обёртка — логика в visual_search.index.benchmark.run_benchmark.
Отвечает на три вопроса: алгоритм ANN, квантование, pooling (title/mean).

Запуск (на подвыборке каталога для скорости):
    python scripts/benchmark_index.py \
        --checkpoint experiments/<run_id>/checkpoints/best.pt \
        --images-csv data/raw/dataset_1M/images.csv \
        --val-csv src/visual_search/evaluation/val_dataset/val_dataset.csv \
        --sample-items 50000

Текстовые/мультимодальные режимы требуют токенайзера (data/tokenization); пока
он не готов, оцениваются только image-запросы.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.seed import set_seed
from visual_search.common.io import PROCESSED_DIR, RAW_DIR, INTERIM_DIR
from visual_search.index.benchmark import (
    DEFAULT_BACKENDS,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Путь к чекпойнту (.pt)")
    parser.add_argument("--images-csv", required=True, help="images.csv каталога")
    parser.add_argument("--val-csv", required=True, help="val_dataset.csv")
    parser.add_argument("--out", default=PROCESSED_DIR / "index_benchmark")
    parser.add_argument("--images-root", default=RAW_DIR / "dataset_1M")
    parser.add_argument("--valid-ids", default=INTERIM_DIR / "valid_image_ids.csv", help="valid_image_ids.csv (фильтр из EDA)")
    parser.add_argument("--poolings", nargs="+", default=["title", "mean"], choices=["title", "mean"])
    parser.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    parser.add_argument("--modes", nargs="+", default=["image", "txt", "multimodal"])
    parser.add_argument("--sample-items", type=int, default=None, help="Подвыборка каталога (None = весь)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=False)

    run_benchmark(
        checkpoint=args.checkpoint,
        images_csv=args.images_csv,
        val_csv=args.val_csv,
        out_dir=args.out,
        images_root=args.images_root,
        valid_ids_csv=args.valid_ids,
        poolings=tuple(args.poolings),
        backends=tuple(args.backends),
        modes=tuple(args.modes),
        sample_items=args.sample_items,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        tokenize=None,  # подключить, когда будет data/tokenization
    )


if __name__ == "__main__":
    main()
