"""torch Dataset поверх train.parquet (контракт §5.1).

Отдаёт пары (изображение, текст) для контрастивного обучения. Читает только
зафиксированные колонки train.parquet; пути к картинкам — относительные от
data/raw/dataset_1M/.

TODO(Подготовка данных): __getitem__ -> (image_tensor, tokens) + выбор query.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class ProductPairDataset(Dataset):
    """Пары image-text для contrastive-обучения."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError
