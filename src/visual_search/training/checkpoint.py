"""Сохранение/загрузка чекпойнтов в experiments/<run_id>/checkpoints/.

Чекпойнт несёт веса + имя модели/конфиг, чтобы build_model мог восстановить
точно ту же архитектуру при оценке и в сервисе.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from visual_search.common.logging import get_logger

log = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    config: DictConfig,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": OmegaConf.to_container(config, resolve=True),
        },
        path,
    )
    log.info("Saved checkpoint: %s (epoch=%d, step=%d)", path, epoch, global_step)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    log.info(
        "Loaded checkpoint: %s (epoch=%d, step=%d)",
        path,
        state["epoch"],
        state["global_step"],
    )
    return state
