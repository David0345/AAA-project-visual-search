"""
Контрастивный лосс: InfoNCE (CLIP).
Притягивает эмбеддинги запроса и позитива, отталкивает негативы ->
единое векторное пространство картинок и текста.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (Contrastive) Loss.
    Требует, чтобы image_embeds и text_embeds были L2-нормированы.
    """
    def __init__(self, init_temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        # logit_scale = log(1 / temperature)
        init_logit_scale = math.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale), requires_grad=learnable)

        self.max_logit_scale = math.log(100.0)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        image_embeds: (B, D), L2-normed
        text_embeds: (B, D), L2-normed
        """
        logit_scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()

        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image_embeds.device)

        loss_i2t = self.ce_loss(logits_per_image, labels)
        loss_t2i = self.ce_loss(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2.0
