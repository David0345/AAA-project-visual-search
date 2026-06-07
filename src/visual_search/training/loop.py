"""Epoch/step логика: forward, лосс, backward, шаг оптимизатора, валидация.

Ожидаемый формат батча от DataLoader (контракт с collate.py):
    {
        "images": Tensor(B, C, H, W),
        "input_ids": Tensor(B, L),
        "attention_mask": Tensor(B, L),
        "item_ids": List[int]
    }
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from visual_search.common.logging import get_logger
from visual_search.training.checkpoint import save_checkpoint
from visual_search.training.tracking import MetricsTracker

log = get_logger(__name__)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: DictConfig,
    tracker: MetricsTracker,
    epoch: int,
    global_step: int,
    scaler: GradScaler | None = None,
) -> tuple[float, int]:
    model.train()
    accum = config.train.gradient_accumulation
    max_norm = config.train.max_grad_norm
    log_every = config.train.get("log_every", 50)
    ckpt_every = config.train.checkpoint_every
    use_amp = scaler is not None

    running_loss = 0.0
    n_optim_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for micro_step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(input_ids, attention_mask=attention_mask)
            loss = loss_fn(img_emb, txt_emb) / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum_boundary = (micro_step + 1) % accum == 0
        is_last = micro_step == len(loader) - 1
        if not (is_accum_boundary or is_last):
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        n_optim_steps += 1
        step_loss = loss.item() * accum
        running_loss += step_loss

        if global_step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            tracker.log(
                global_step,
                {"epoch": epoch, "train_loss": round(step_loss, 6), "lr": lr},
            )
            log.info(
                "epoch %d | step %d | loss %.4f | lr %.2e",
                epoch,
                global_step,
                step_loss,
                lr,
            )

        if ckpt_every and global_step % ckpt_every == 0:
            ckpt_path = (
                Path(config.output_dir) / "checkpoints" / f"step_{global_step}.pt"
            )
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, epoch, global_step, config
            )

    scheduler.step()
    avg_loss = running_loss / max(n_optim_steps, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(input_ids)
        total_loss += loss_fn(img_emb, txt_emb).item()
        n += 1
    return total_loss / max(n, 1)
