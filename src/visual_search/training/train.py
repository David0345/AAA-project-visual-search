"""
Оркестрация прогона: конфиг -> данные + модель + loop -> чекпойнт.
Это логика; запускается тонкой обёрткой scripts/train.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from visual_search.common.logging import get_logger
from visual_search.models import build_model
from visual_search.training.checkpoint import load_checkpoint, save_checkpoint
from visual_search.training.loop import train_one_epoch, validate
from visual_search.training.optim import build_optimizer, build_scheduler
from visual_search.training.tracking import MetricsTracker
from visual_search.data.datamodule import build_dataloaders

log = get_logger(__name__)


def run(config: DictConfig, device: torch.device) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(config))

    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    model = build_model(model_cfg)
    model = model.to(device)
    log.info("Model %s (embed_dim=%d) on %s", config.model.name, model.embed_dim, device)

    train_loader, val_loader = build_dataloaders(config)

    loss_fn = instantiate(config.train.loss).to(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    use_amp = config.train.get("amp", device.type == "cuda")
    scaler = GradScaler(device.type) if use_amp and device.type == "cuda" else None

    output_dir = Path(config.output_dir)
    tracker = MetricsTracker(output_dir)

    start_epoch = 0
    global_step = 0
    resume = config.train.get("resume_from")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    best_val = float("inf")
    avg_loss = None
    log.info("Training: %d epochs, starting from epoch %d", config.train.epochs, start_epoch)

    for epoch in range(start_epoch, config.train.epochs):
        avg_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            tracker=tracker,
            epoch=epoch,
            global_step=global_step,
            scaler=scaler,
        )
        log.info("Epoch %d: avg_train_loss=%.4f", epoch, avg_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, loss_fn, device)
            log.info("Epoch %d: val_loss=%.4f", epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model, optimizer, scheduler, epoch, global_step, config,
                )

        save_checkpoint(
            output_dir / "checkpoints" / f"epoch_{epoch}.pt",
            model, optimizer, scheduler, epoch, global_step, config,
        )

    if avg_loss is not None:
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model, optimizer, scheduler, config.train.epochs - 1, global_step, config,
        )
    else:
        log.warning("No epochs to train (start_epoch=%d >= epochs=%d)", start_epoch, config.train.epochs)

    tracker.log_summary({
        "total_steps": global_step,
        "epochs_completed": config.train.epochs - start_epoch,
        "final_train_loss": round(avg_loss, 6) if avg_loss is not None else None,
        "best_val_loss": round(best_val, 6) if best_val != float("inf") else None,
    })
    tracker.close()
    log.info("Done. Artifacts saved to %s", output_dir)
