"""Сборка оптимизатора и LR-scheduler из train-секции конфига.

Используется Hydra instantiate — _target_ резолвится из YAML.
"""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def build_optimizer(
    config: DictConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    return instantiate(config.train.optimizer, params=model.parameters())


def build_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer):
    return instantiate(config.train.scheduler, optimizer=optimizer)
