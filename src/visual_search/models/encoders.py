"""
Реализации энкодеров — обёртки над transformers (CLIP).
Каждая реализация удовлетворяет контракту base.Encoder и регистрируется в registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from visual_search.models.registry import register


@register("clip_vit_b32")
@register("ruclip_vit_b32")
class RuCLIPEncoder(nn.Module):
    """
    Обёртка над ai-forever/ruclip-vit-base-patch32-224 из HuggingFace.
    Возвращает L2-нормированные эмбеддинги размерности 512.
    """
    def __init__(self, config: dict):
        super().__init__()
        pretrained_name = config.get("pretrained", "ai-forever/ruclip-vit-base-patch32-224")
        self.model = CLIPModel.from_pretrained(pretrained_name)

        self.embed_dim = self.model.config.projection_dim

        # Флаг для заморозки весов (для zero-shot baseline)
        self.freeze = config.get("freeze", False)
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) -> (B, embed_dim), L2-norm."""
        img_embeds = self.model.get_image_features(pixel_values=images)
        if hasattr(img_embeds, 'pooler_output'):
            img_embeds = img_embeds.pooler_output
        return F.normalize(img_embeds, p=2, dim=-1)

    def encode_text(self, tokens: torch.Tensor,
                    attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, L), attention_mask: (B, L) -> (B, embed_dim), L2-norm."""
        kwargs = {"input_ids": tokens}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        txt_embeds = self.model.get_text_features(**kwargs)
        if hasattr(txt_embeds, 'pooler_output'):
            txt_embeds = txt_embeds.pooler_output
        return F.normalize(txt_embeds, p=2, dim=-1)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor,
                attention_mask: torch.Tensor | None = None):
        """Опциональный метод, если нужно закодировать обе модальности сразу."""
        return self.encode_image(images), self.encode_text(tokens, attention_mask)
