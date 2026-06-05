"""Factory для создания DataLoader'ов из Hydra-конфига."""

from __future__ import annotations

import logging
from typing import Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from visual_search.data.dataset import ContrastiveImageTextDataset, SearchEvalDataset
from visual_search.data.collate import contrastive_collate_fn, eval_collate_fn
from visual_search.models.registry import get_processor

log = logging.getLogger(__name__)


def create_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создаёт train/val/test DataLoader'ы из конфига.

    Args:
        config: Hydra-конфиг с секциями data, model, seed

    Returns:
        train_loader, val_loader, test_loader
    """
    processor = get_processor(config.model.pretrained)

    train_ds = ContrastiveImageTextDataset(
        parquet_path=config.data.train_path,
        image_root=config.data.image_root,
        processor=processor,
        seed=config.seed.seed if hasattr(config, 'seed') else 42,
        max_queries_per_item=config.data.get('max_queries_per_item', None),
    )

    val_ds = SearchEvalDataset(
        csv_path=config.data.val_path,
        image_root=config.data.image_root,
        processor=processor,
    )

    test_ds = SearchEvalDataset(
        csv_path=config.data.test_path,
        image_root=config.data.image_root,
        processor=processor,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=contrastive_collate_fn,
        drop_last=True,  # для стабильного batch norm / contrastive loss
    )

    # для eval: batch_size=1, чтобы не смешивать разные модальности
    eval_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=eval_collate_fn,
    )

    val_loader = DataLoader(val_ds, **eval_kwargs)
    test_loader = DataLoader(test_ds, **eval_kwargs)

    log.info(f'DataLoaders created: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}')

    return train_loader, val_loader, test_loader
