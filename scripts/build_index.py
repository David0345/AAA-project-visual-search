"""CLI: сборка ANN-индекса из эмбеддингов каталога.

Тонкая обёртка — логика в visual_search.index.build_index.
Запуск:
    python scripts/build_index.py \
        --checkpoint experiments/<run_id>/checkpoints/best.pt \
        --images-csv data/raw/dataset_1M/images.csv \
        --out data/processed/index
"""

from __future__ import annotations

import argparse

from visual_search.index.build_index import build_index

from visual_search.common.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Путь к чекпойнту (.pt)")
    parser.add_argument("--images-csv", required=True, help="images.csv: image_id,item_id,image_path,is_title")
    parser.add_argument("--out", default="data/processed/index", help="Директория для индекса")
    parser.add_argument("--images-root", default="data/raw/dataset_1M", help="Корень изображений")
    parser.add_argument("--valid-ids", default=None, help="valid_image_ids.csv (фильтр из EDA)")
    parser.add_argument("--pooling", default="title", choices=["title", "mean"])
    parser.add_argument("--backend", default="flat", choices=["flat", "ivf", "ivfpq", "hnsw"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    build_index(
        checkpoint=args.checkpoint,
        images_csv=args.images_csv,
        out_dir=args.out,
        images_root=args.images_root,
        valid_ids_csv=args.valid_ids,
        pooling=args.pooling,
        backend=args.backend,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
