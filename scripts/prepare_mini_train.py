#!/usr/bin/env python3
"""Сборка mini_train.parquet из локальных изображений для smoke-тест файн-тюнинга.

Использует tmp_manifest_with_urls.csv (метаданные товаров) + images.csv (пути
к картинкам) из архива датасета. Применяет фильтр EDA (valid_image_ids.csv).
Оставляет только товары, чьи изображения есть локально.

Запуск:
    python scripts/prepare_mini_train.py
    # или с явными путями:
    python scripts/prepare_mini_train.py \
        --data-dir data/raw/dataset_1M \
        --valid-ids src/visual_search/data/eda/valid_image_ids.csv \
        --output data/interim/mini_train.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import RAW_DIR, INTERIM_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Импортируем генераторы запросов из build_train.py
from visual_search.data.prepare.category_synonyms import CATEGORY_SYNONYMS
from visual_search.data.prepare.brand_translit import BRAND_RU
from visual_search.data.prepare.build_train import (
    generate_product_text,
    generate_queries,
    clean_brand,
)


def collect_local_image_ids(images_dir: Path) -> set[str]:
    """Собираем имена файлов (без .jpg) локально доступных изображений."""
    log.info("Сканируем локальные изображения в %s ...", images_dir)
    storage_ids: set[str] = set()
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.endswith(".jpg"):
                storage_ids.add(f[:-4])  # image_storage_image_id (без расширения)
    log.info("  Локально: %d jpg-файлов", len(storage_ids))
    return storage_ids


def build_items_from_manifest(
    manifest_path: Path,
    valid_item_ids: set[int],
) -> pd.DataFrame:
    """Читаем tmp_manifest_with_urls.csv и берём уникальные товары."""
    log.info("Читаем манифест (может занять ~20 сек) ...")
    cols = [
        "item_id", "predmet_odezhdy", "param2",
        "cvet", "brand", "sostoyanie", "category_name",
    ]
    # Читаем чанками чтобы не грузить 500 МБ целиком
    chunks = []
    for chunk in pd.read_csv(manifest_path, usecols=cols, chunksize=100_000):
        chunk = chunk[chunk["item_id"].isin(valid_item_ids)]
        chunks.append(chunk)

    manifest = pd.concat(chunks, ignore_index=True)
    items = manifest.drop_duplicates("item_id").reset_index(drop=True)
    log.info("  Уникальных товаров в манифесте (из локальных): %d", len(items))
    return items


def prepare(
    data_dir: Path,
    valid_ids_path: Path,
    output_path: Path,
    images_subdir: str = "images",
) -> None:
    images_dir = data_dir / images_subdir

    # 1. Локальные файлы
    local_storage_ids = collect_local_image_ids(images_dir)

    # 2. images.csv → image_id, image_storage_image_id, image_path, is_title
    log.info("Загружаем images.csv ...")
    images_df = pd.read_csv(
        data_dir / "images.csv",
        usecols=["item_id", "image_id", "image_storage_image_id", "image_path", "is_title"],
    )
    log.info("  images.csv: %d строк", len(images_df))

    # 3. EDA-фильтр
    log.info("Загружаем valid_image_ids.csv ...")
    valid_df = pd.read_csv(valid_ids_path)
    valid_image_ids = set(valid_df["image_id"].tolist())
    images_df = images_df[images_df["image_id"].isin(valid_image_ids)]
    log.info("  После EDA-фильтра: %d строк", len(images_df))

    # 4. Оставляем только локально доступные файлы
    images_df["storage_id_str"] = images_df["image_storage_image_id"].astype(str)
    images_df = images_df[images_df["storage_id_str"].isin(local_storage_ids)]
    log.info("  После фильтра локальных файлов: %d строк", len(images_df))

    valid_item_ids = set(images_df["item_id"].tolist())
    log.info("  Уникальных item_id: %d", len(valid_item_ids))

    # 5. Метаданные из манифеста
    items_df = build_items_from_manifest(data_dir / "tmp_manifest_with_urls.csv", valid_item_ids)

    # 6. Титульные и дополнительные изображения
    log.info("Собираем title/other изображения ...")
    title_images = (
        images_df[images_df["is_title"] == True]
        .groupby("item_id")["image_path"]
        .first()
        .reset_index(name="title_image_path")
    )
    other_images = (
        images_df[images_df["is_title"] == False]
        .groupby("item_id")["image_path"]
        .apply(list)
        .reset_index(name="other_image_paths")
    )

    # Если у товара нет is_title=True, берём любую первую
    no_title = set(valid_item_ids) - set(title_images["item_id"])
    if no_title:
        fallback = (
            images_df[images_df["item_id"].isin(no_title)]
            .groupby("item_id")["image_path"]
            .first()
            .reset_index(name="title_image_path")
        )
        title_images = pd.concat([title_images, fallback], ignore_index=True)

    train_df = items_df.merge(title_images, on="item_id", how="inner")
    train_df = train_df.merge(other_images, on="item_id", how="left")
    train_df["other_image_paths"] = train_df["other_image_paths"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    log.info("  После join с изображениями: %d товаров", len(train_df))

    # 7. Генерируем product_text и queries
    log.info("Генерируем product_text и queries ...")
    train_df["product_text"] = train_df.apply(generate_product_text, axis=1)
    train_df["queries"] = train_df.apply(generate_queries, axis=1)

    # Убираем товары без запросов (категория "Другое" и т.п.)
    train_df = train_df[train_df["queries"].map(len) > 0].reset_index(drop=True)
    log.info("  После фильтра пустых queries: %d товаров", len(train_df))

    # 8. Финальные колонки
    final_cols = [
        "item_id", "title_image_path", "other_image_paths",
        "product_text", "queries",
        "predmet_odezhdy", "param2", "cvet", "brand", "sostoyanie", "category_name",
    ]
    output_df = train_df[final_cols]

    # 9. Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    log.info("Сохранено: %s  (%d строк)", output_path, len(output_df))

    # Статистика
    log.info("--- Статистика ---")
    log.info("  Товаров: %d", len(output_df))
    log.info("  Avg queries per item: %.1f",
             output_df["queries"].map(len).mean())
    log.info("  Категории: %s",
             output_df["param2"].value_counts().head(5).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path,
                        default=RAW_DIR / "dataset_1M")
    parser.add_argument("--valid-ids", type=Path,
                        default=INTERIM_DIR / "valid_image_ids.csv")
    parser.add_argument("--output", type=Path,
                        default=INTERIM_DIR / "mini_train.parquet")
    args = parser.parse_args()

    prepare(
        data_dir=args.data_dir,
        valid_ids_path=args.valid_ids,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
