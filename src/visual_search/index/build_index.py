"""Сборка ANN-индекса из эмбеддингов каталога и сохранение на диск.

Это логика; запускается тонкой обёрткой scripts/build_index.py. Результат —
в data/processed/<name>/ (индекс + маппинг id), переиспользуется оценкой и
сервисом.

Поток: чекпойнт -> модель -> embed_catalog -> ANNIndex.build -> save.
Стратегию (pooling) и backend берём из аргументов; оптимальные значения
подбирает scripts/benchmark_index.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from visual_search.common.logging import get_logger
from visual_search.index.ann import ANNIndex, IndexSpec
from visual_search.index.embed import embed_catalog
from visual_search.models import build_model

log = get_logger(__name__)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model_from_checkpoint(checkpoint: str | Path, device: torch.device):
    """Восстановить Encoder из чекпойнта: build_model(config.model) + веса."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = state["config"]["model"]
    model = build_model(model_cfg)
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()
    log.info("Модель восстановлена из %s (%s, embed_dim=%d)", checkpoint, model_cfg.get("name"), model.embed_dim)
    return model


def load_images_csv(
    images_csv: str | Path, valid_ids_csv: str | Path | None = None
) -> pd.DataFrame:
    """Прочитать images.csv (image_id, item_id, image_path, is_title).

    Опционально оставить только валидные изображения (из EDA-фильтрации:
    valid_image_ids.csv со столбцами image_id, item_id).
    """
    df = pd.read_csv(images_csv)
    if valid_ids_csv is not None:
        valid = pd.read_csv(valid_ids_csv)
        before = len(df)
        df = df.merge(valid[["image_id", "item_id"]], on=["image_id", "item_id"], how="inner")
        log.info("Фильтрация по valid_ids: %d -> %d изображений", before, len(df))
    return df


def build_index(
    checkpoint: str | Path,
    images_csv: str | Path,
    out_dir: str | Path,
    *,
    images_root: str | Path = "data/raw/dataset_1M",
    valid_ids_csv: str | Path | None = None,
    pooling: str = "title",
    backend: str = "flat",
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "auto",
) -> Path:
    """Полный путь: чекпойнт + каталог -> сохранённый ANN-индекс.

    Returns путь к директории индекса.
    """
    dev = resolve_device(device)
    model = load_model_from_checkpoint(checkpoint, dev)
    images_df = load_images_csv(images_csv, valid_ids_csv)

    catalog = embed_catalog(
        model,
        images_df,
        images_root=images_root,
        pooling=pooling,
        batch_size=batch_size,
        device=dev,
        image_size=image_size,
        num_workers=num_workers,
    )

    index = ANNIndex(embed_dim=model.embed_dim, spec=IndexSpec(backend=backend))
    index.build(catalog.vectors, catalog.item_ids)

    out = Path(out_dir)
    index.save(out)

    # Мост item_id -> [image_id]: нужен оценке (таргеты заданы в image_id).
    (out / "item_images.json").write_text(
        json.dumps({str(k): v for k, v in catalog.item_to_images.items()}),
        encoding="utf-8",
    )
    (out / "build_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "pooling": pooling,
                "backend": backend,
                "num_items": int(len(catalog.item_ids)),
                "embed_dim": int(model.embed_dim),
                "images_root": str(images_root),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Готово. Индекс и маппинги в %s", out)
    return out
