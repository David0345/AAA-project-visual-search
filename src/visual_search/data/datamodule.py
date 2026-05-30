"""Сборка train/val DataLoader'ов из конфига data-секции.

Единая точка, которую зовёт training/: train_dataloader() / val_dataloader().

TODO(Подготовка данных): собрать Dataset + transforms + collate + DataLoader.
"""

from __future__ import annotations
