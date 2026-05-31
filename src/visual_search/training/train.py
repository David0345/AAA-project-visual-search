"""Оркестрация прогона: конфиг -> данные + модель + loop -> чекпойнт.

Это логика; запускается тонкой обёрткой scripts/train.py.

TODO(Обучение): run(config) — seed, datamodule, build_model, optim, loop, save.
"""

from __future__ import annotations
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch

from typing import Any


def run(config: DictConfig, device: torch.device) -> None:
    model = ...

    # Создание оптимизатора и шедулера напрямую из конфига
    optimizer = instantiate(config.train.optimizer, params=model.parameters())
    scheduler = instantiate(config.train.scheduler, optimizer=optimizer)

    raise NotImplementedError
