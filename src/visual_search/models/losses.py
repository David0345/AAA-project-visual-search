"""Контрастивные лоссы для мультимодального обучения.

InfoNCELoss  — симметричный CLIP-лосс
SigmoidLoss  — SigLIP-лосс, работает лучше при маленьком batch_size

Формулы
-------
InfoNCE:
    logits_i2t = scale * img_emb @ txt_emb.T    (B, B)
    labels = arange(B)
    L = (CE(logits_i2t, labels) + CE(logits_i2t.T, labels)) / 2

Sigmoid (SigLIP):
    L_ij = log σ(z_ij * y_ij * scale)
    y_ij = +1 если i==j, -1 иначе
    Усредняем по всем B² парам

Конфиг:
    train:
      loss:
        _target_: visual_search.models.losses.InfoNCELoss
        temperature: 0.07
        learnable_temperature: true
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Симметричный InfoNCE (CLIP-лосс).

    Args:
        temperature: начальная температура (< 1 → острее распределение).
        learnable_temperature: если True — temperature обучается.
        max_logit_scale: клиппинг exp(log_scale) для стабильности (100 = CLIP-стандарт).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale

        # Храним log(scale) = -log(temperature) для численной стабильности
        log_scale = math.log(1.0 / temperature)
        if learnable_temperature:
            self.log_scale = nn.Parameter(torch.tensor(log_scale))
        else:
            self.register_buffer("log_scale", torch.tensor(log_scale))

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=self.max_logit_scale)

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: (B, D) L2-нормированные
            text_embeds:  (B, D) L2-нормированные

        Returns:
            scalar loss
        """
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)

        # (B, B) матрица сходства
        logits_i2t = self.logit_scale * image_embeds @ text_embeds.T
        logits_t2i = logits_i2t.T

        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)

        return (loss_i2t + loss_t2i) / 2.0

    def extra_repr(self) -> str:
        return f"logit_scale={self.logit_scale.item():.3f}"


class SigmoidLoss(nn.Module):
    """SigLIP-лосс — sigmoid вместо softmax.

    Лучше работает при малом batch_size, т.к. не требует B×B normalizer.
    Рекомендуется при batch_size < 128.

    Args:
        temperature: начальная температура.
        bias: смещение (SigLIP обучает его совместно).
        learnable: обучать ли temperature и bias.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        bias: float = -10.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        log_scale = math.log(1.0 / temperature)
        if learnable:
            self.log_scale = nn.Parameter(torch.tensor(log_scale))
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer("log_scale", torch.tensor(log_scale))
            self.register_buffer("bias", torch.tensor(bias))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: (B, D) L2-нормированные
            text_embeds:  (B, D) L2-нормированные

        Returns:
            scalar loss
        """
        batch_size = image_embeds.size(0)
        scale = self.log_scale.exp()

        # (B, B) логиты
        logits = scale * (image_embeds @ text_embeds.T) + self.bias

        # y_ij = +1 на диагонали, -1 везде
        labels = 2.0 * torch.eye(batch_size, device=logits.device) - 1.0

        # Sigmoid binary cross-entropy по всем парам
        loss = -F.logsigmoid(labels * logits).mean()
        return loss


# ---------------------------------------------------------------------------
# Фабрика для Hydra (_target_)
# ---------------------------------------------------------------------------
# Hydra умеет инстанцировать классы напрямую. Но для удобства — алиасы:
infonce_loss = InfoNCELoss
sigmoid_loss = SigmoidLoss
