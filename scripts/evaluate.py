"""CLI: оценка модели на val_dataset.

Тонкая обёртка — логика в visual_search.evaluation.evaluate.

Примеры::

    # Полный прогон (модель + индекс)
    python scripts/evaluate.py \\
        --checkpoint experiments/baseline/checkpoint.pt \\
        --index      experiments/baseline/index.bin \\
        --images-base data/raw/dataset_1M

    # Только метрики без рисков загрузки модели (search_fn из файла)
    python scripts/evaluate.py \\
        --val-csv src/visual_search/evaluation/val_dataset/val_dataset.csv \\
        --dry-run   # выведет статистику датасета и выйдет
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="Путь к .pt-файлу чекпоинта модели")
    parser.add_argument("--index", default=None, help="Путь к файлу ANN-индекса")
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Путь к val_dataset.csv (по умолчанию — встроенный в пакет)",
    )
    parser.add_argument(
        "--images-base",
        default="",
        help="Префикс для путей изображений (например, data/raw/dataset_1M)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        metavar="K",
        help="Список K для Recall@K и Precision@K (по умолчанию: 1 5 10)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Сохранить результаты в JSON-файл",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только загрузить датасет, распечатать статистику и выйти",
    )
    args = parser.parse_args()

    from visual_search.evaluation.val_dataset import ValDataset
    from visual_search.evaluation.evaluate import print_report

    # --- dry-run: просто показать статистику датасета ---
    if args.dry_run:
        ds_kwargs = {"csv_path": args.val_csv} if args.val_csv else {}
        ds = ValDataset(**ds_kwargs, images_base=args.images_base)
        st = ds.stats()
        print(f"ValDataset: total={st['total']}  image={st['image']}  txt={st['txt']}  multimodal={st['multimodal']}")
        return

    # --- Полный прогон ---
    if args.checkpoint is None or args.index is None:
        parser.error("--checkpoint и --index обязательны для полного прогона (или используйте --dry-run)")

    from visual_search.models.registry import build_model  # noqa: F401 — TODO заменить на load_checkpoint
    from visual_search.evaluation.evaluate import evaluate

    # TODO(Оценка): реализовать load_checkpoint(path) → model после того, как
    #               training/loop.py научится сохранять чекпоинты.
    raise NotImplementedError(
        "load_checkpoint ещё не реализован — ждём PR от команды Обучения.\n"
        "Для тестирования метрик используйте evaluate_with_search_fn напрямую."
    )

    # Код ниже будет активирован после появления load_checkpoint:
    # model = load_checkpoint(args.checkpoint)
    # index = ANNIndex.load(args.index)
    # results = evaluate(
    #     model, index,
    #     dataset_path=args.val_csv,
    #     images_base=args.images_base,
    #     k_values=args.k_values,
    # )
    # print_report(results)
    # if args.output_json:
    #     with open(args.output_json, "w") as f:
    #         json.dump({mode: m.as_flat_dict() for mode, m in results.items()}, f, indent=2, ensure_ascii=False)
    #     logger.info("Результаты сохранены в %s", args.output_json)


if __name__ == "__main__":
    main()
