#!/usr/bin/env python3
"""
Скрипт для сборки тренировочного датасета train.parquet для fine-tuning RuCLIP.

Использует результаты EDA и фильтрации (valid_image_ids.csv),
словари синонимов категорий и транслитераций брендов.

Пример запуска:
    python prepare_train_data.py \
        --data-dir dataset_1M \
        --valid-ids valid_image_ids.csv \
        --output train.parquet
"""

import argparse
import re
from pathlib import Path
import pandas as pd

from category_synonyms import CATEGORY_SYNONYMS
from brand_translit import BRAND_RU


def load_and_filter_data(data_dir: Path, valid_ids_path: Path):
    """Загружает items.csv, images.csv и оставляет только валидные изображения."""
    items = pd.read_csv(data_dir / "items.csv")
    images = pd.read_csv(data_dir / "images.csv")
    valid_ids = pd.read_csv(valid_ids_path)  # колонки image_id, item_id

    # INNER JOIN оставляет только строки, прошедшие фильтрацию
    images = images.merge(valid_ids, on=["image_id", "item_id"], how="inner")

    # Оставляем только те объявления, для которых есть хотя бы одно изображение
    valid_item_ids = images["item_id"].unique()
    items = items[items["item_id"].isin(valid_item_ids)]

    return items, images


def get_title_image_path(group_df):
    """Для группы изображений одного товара возвращает путь к титульному изображению.
    Если титульного нет, берётся первое доступное."""
    titles = group_df[group_df.is_title == True].image_path
    if len(titles) > 0:
        return titles.iloc[0]
    return group_df.image_path.iloc[0]


def add_image_columns(items, images):
    """Добавляет в датафрейм items колонки с путями к титульному и дополнительным изображениям."""
    title_df = (
        images.groupby('item_id', group_keys=False)
              .apply(get_title_image_path, include_groups=False)
              .reset_index(name='title_image_path')
    )

    other_df = (
        images[images.is_title == False]
        .groupby('item_id')['image_path']
        .apply(list)
        .reset_index(name='other_image_paths')
    )

    train_df = items.merge(title_df, on='item_id', how='inner')
    train_df = train_df.merge(other_df, on='item_id', how='left')

    train_df['other_image_paths'] = train_df['other_image_paths'].apply(
        lambda d: d if isinstance(d, list) else []
    )
    return train_df


def generate_product_text(row):
    """Формирует описание товара в виде перечисления через запятую.
    Пример: 'Пиджаки и костюмы, Чёрный, Stradivarius', состояние: Отличное."""
    parts = []
    parts.append(str(row['predmet_odezhdy']))
    if pd.notna(row['cvet']):
        parts.append(str(row['cvet']))
    if pd.notna(row['brand']):
        parts.append(str(row['brand']))
    if pd.notna(row['sostoyanie']):
        parts.append(f"состояние: {row['sostoyanie']}")
    return ', '.join(parts)


def clean_brand(raw_brand: str | None) -> str | None:
    """Очищает название бренда от небуквенных символов, приводит к нижнему регистру."""
    if pd.isna(raw_brand) or raw_brand is None:
        return None
    brand = str(raw_brand).lower()
    brand = re.sub(r'[^a-zа-яё\s]', '', brand)
    brand = brand.strip()
    return brand if brand else None


def generate_queries(row):
    """Генерирует поисковые запросы на основе атрибутов товара.
    Используются естественные синонимы категорий (например, 'пиджак' вместо 'Пиджаки и костюмы'),
    а также комбинации с цветом и брендом (включая русские транслитерации).
    """
    predmet_orig = str(row['predmet_odezhdy']).strip()
    if predmet_orig == 'Другое':
        return []

    bases = CATEGORY_SYNONYMS.get(predmet_orig, [predmet_orig.lower()])

    cvet = row['cvet'] if pd.notna(row['cvet']) else None

    brand_raw = row['brand'] if pd.notna(row['brand']) else None
    brand_clean = clean_brand(brand_raw) if brand_raw else None
    if brand_clean in ('без бренда', 'другой'):
        brand_clean = None
    brand_ru = BRAND_RU.get(brand_clean) if brand_clean else None

    queries = set()

    for base in bases:
        queries.add(base)

        if cvet:
            queries.add(f'{cvet} {base}')
            queries.add(f'{base} {cvet}')

        if brand_clean:
            queries.add(f'{brand_clean} {base}')
            queries.add(f'{base} {brand_clean}')
        if brand_ru:
            queries.add(f'{brand_ru} {base}')
            queries.add(f'{base} {brand_ru}')

        if cvet and brand_clean:
            for combo in [
                f'{cvet} {brand_clean} {base}',
                f'{brand_clean} {cvet} {base}',
                f'{base} {cvet} {brand_clean}',
                f'{base} {brand_clean} {cvet}'
            ]:
                queries.add(combo)
        if cvet and brand_ru:
            for combo in [
                f'{cvet} {brand_ru} {base}',
                f'{brand_ru} {cvet} {base}',
                f'{base} {cvet} {brand_ru}',
                f'{base} {brand_ru} {cvet}'
            ]:
                queries.add(combo)

    queries = {q for q in queries if 1 <= len(q.split()) <= 6 and q.strip()}
    return list(queries)


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', type=Path, required=True, help='Путь к папке с датасетом'
    )
    parser.add_argument(
        '--valid-ids', type=Path, required=True, help='Путь к valid_image_ids.csv (от EDA-фильтрации)',
    )
    parser.add_argument(
        '--output', default='train.parquet', type=Path, help='Путь для сохранения выходного parquet',
    )
    args = parser.parse_args()

    print('Загрузка и фильтрация данных...')
    items, images = load_and_filter_data(args.data_dir, args.valid_ids)
    print(f'Объявлений после фильтрации: {len(items)}')

    print("Сборка колонок с изображениями...")
    train_df = add_image_columns(items, images)

    print("Генерация product_text...")
    train_df["product_text"] = train_df.apply(generate_product_text, axis=1)

    print("Генерация queries...")
    train_df["queries"] = train_df.apply(generate_queries, axis=1)

    final_columns = [
        'item_id',
        'title_image_path',
        'other_image_paths',
        'product_text',
        'queries',
        'predmet_odezhdy',
        'param2',
        'cvet',
        'brand',
        'sostoyanie',
        'category_name',
    ]
    output_df = train_df[final_columns]

    print(f'Сохранение в {args.output}...')
    output_df.to_parquet(args.output, index=False)


if __name__ == "__main__":
    prepare()
