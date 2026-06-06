"""Батч-инференс эмбеддингов каталога обученной моделью.

Каталог индексируется на уровне ТОВАРА: один вектор на item_id. Стратегия
агрегации изображений товара выбирается параметром ``pooling``:

  * ``"title"`` — берём только титульное изображение товара (is_title);
  * ``"mean"``  — усредняем эмбеддинги ВСЕХ изображений товара и заново
                  L2-нормируем (среднее нормированных векторов не нормировано).

Выход: (N_items, embed_dim) float32 L2-norm + параллельный массив item_id
(контракт §5.4). Какая стратегия лучше — решает scripts/benchmark_index.py.

Источник каталога — images.csv со столбцами: image_id, item_id, image_path,
is_title (как в data/prepare/build_train.py). Пути к картинкам — относительные
от images_root (по умолчанию data/raw/dataset_1M/).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from visual_search.common.logging import get_logger

log = get_logger(__name__)

# Нормализация как у CLIP (open_clip / openai). Должна совпадать с
# препроцессингом, на котором обучалась модель.
_CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def default_image_transform(image_size: int = 224) -> Callable:
    """CLIP-препроцессинг для inference (resize -> center crop -> normalize)."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(_CLIP_MEAN, _CLIP_STD),
        ]
    )


class _ImagePathDataset(Dataset):
    """Отдаёт (tensor, ok) по списку путей; битые/отсутствующие -> (zeros, False)."""

    def __init__(self, paths: list[str], images_root: Path, transform: Callable, image_size: int):
        self._paths = paths
        self._root = images_root
        self._transform = transform
        self._size = image_size

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        path = self._root / self._paths[idx]
        try:
            with Image.open(path) as img:
                tensor = self._transform(img.convert("RGB"))
            return tensor, True
        except (OSError, ValueError) as exc:
            log.warning("Не удалось прочитать %s: %s", path, exc)
            return torch.zeros(3, self._size, self._size), False


@torch.no_grad()
def embed_images(
    model,
    image_paths: list[str],
    *,
    images_root: str | Path,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
    transform: Callable | None = None,
    image_size: int = 224,
    num_workers: int = 0,
) -> np.ndarray:
    """list[str] относительных путей -> (len, embed_dim) float32, L2-norm.

    Ядро инференса — переиспользуется и для каталога, и для запросных картинок.
    Эмбеддинги битых картинок зануляются (никогда не матчатся в IP-поиске).
    """
    transform = transform or default_image_transform(image_size)
    device = torch.device(device)
    model.eval()

    dataset = _ImagePathDataset(image_paths, Path(images_root), transform, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    chunks: list[np.ndarray] = []
    for images, ok in loader:
        images = images.to(device, non_blocking=True)
        emb = model.encode_image(images).float()
        emb[~ok.to(device)] = 0.0  # зануляем битые
        chunks.append(emb.cpu().numpy())

    if not chunks:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


@dataclass
class CatalogEmbeddings:
    """Каталог на уровне товара + мост item_id -> его image_id (для оценки)."""

    vectors: np.ndarray  # (N_items, embed_dim) float32 L2-norm
    item_ids: np.ndarray  # (N_items,) int64, параллелен vectors
    item_to_images: dict[int, list[int]]


def _select_title_rows(images_df: pd.DataFrame) -> pd.DataFrame:
    """По одной строке на товар: титульное изображение, иначе первое доступное."""
    df = images_df.sort_values("is_title", ascending=False)
    return df.groupby("item_id", as_index=False, sort=False).first()


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _mean_pool_by_item(
    vectors: np.ndarray, item_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Среднее векторов по item_id. Возвращает (means, uniq_item_ids asc)."""
    order = np.argsort(item_ids, kind="stable")
    sorted_ids = item_ids[order]
    sorted_vecs = vectors[order]
    uniq, start = np.unique(sorted_ids, return_index=True)
    sums = np.add.reduceat(sorted_vecs, start, axis=0)
    counts = np.diff(np.append(start, len(sorted_ids)))
    means = sums / counts[:, None]
    return means.astype(np.float32), uniq


def embed_catalog(
    model,
    images_df: pd.DataFrame,
    *,
    images_root: str | Path,
    pooling: str = "title",
    batch_size: int = 256,
    device: torch.device | str = "cpu",
    transform: Callable | None = None,
    image_size: int = 224,
    num_workers: int = 0,
) -> CatalogEmbeddings:
    """Эмбеддинги каталога на уровне товара.

    images_df: столбцы image_id, item_id, image_path, is_title.
    pooling: "title" | "mean".
    """
    required = {"image_id", "item_id", "image_path", "is_title"}
    missing = required - set(images_df.columns)
    if missing:
        raise ValueError(f"images_df: нет столбцов {missing}")

    item_to_images = (
        images_df.groupby("item_id")["image_id"].apply(lambda s: [int(x) for x in s]).to_dict()
    )

    common = dict(
        images_root=images_root,
        batch_size=batch_size,
        device=device,
        transform=transform,
        image_size=image_size,
        num_workers=num_workers,
    )

    if pooling == "title":
        rows = _select_title_rows(images_df)
        log.info("Каталог (title): %d товаров", len(rows))
        vectors = embed_images(model, rows["image_path"].tolist(), **common)
        item_ids = rows["item_id"].to_numpy(dtype=np.int64)
    elif pooling == "mean":
        log.info("Каталог (mean): %d изображений, %d товаров", len(images_df), images_df["item_id"].nunique())
        per_image = embed_images(model, images_df["image_path"].tolist(), **common)
        vectors, item_ids = _mean_pool_by_item(per_image, images_df["item_id"].to_numpy(dtype=np.int64))
        vectors = _l2_normalize(vectors)  # среднее нормированных != нормировано
    else:
        raise ValueError(f"pooling должен быть 'title' | 'mean', получено {pooling!r}")

    return CatalogEmbeddings(vectors=vectors, item_ids=item_ids, item_to_images=item_to_images)
