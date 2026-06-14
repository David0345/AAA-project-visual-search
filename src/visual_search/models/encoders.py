"""Реализации энкодеров — обёртки над open_clip (CLIP, SigLIP, multilingual).

Каждая реализация удовлетворяет контракту base.Encoder и регистрируется в
registry через @register("имя_из_конфига").

Поддерживаемые архитектуры
---------------------------
clip_vit_b32        openai/ViT-B-32          EN,  embed_dim=512
clip_vit_b16        openai/ViT-B-16          EN,  embed_dim=512
clip_vit_l14        openai/ViT-L-14          EN,  embed_dim=768
xlm_clip_vit_b32    laion/xlm-roberta        RU!, embed_dim=512

Выбор модели для Avito
-----------------------
Данные на русском => рекомендуется xlm_clip_vit_b32 как стартовая точка
(pretrained на laion5B, text tower = xlm-roberta-base, знает русский).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_search.models.base import Encoder
from visual_search.models.registry import register

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Маппинг имён конфига -> (open_clip model_name, pretrained tag)
# ---------------------------------------------------------------------------

_MODEL_MAP: dict[str, tuple[str, str]] = {
    "clip_vit_b32":     ("ViT-B-32",  "openai"),
    "clip_vit_b16":     ("ViT-B-16",  "openai"),
    "clip_vit_l14":     ("ViT-L-14",  "openai"),
    # Multilingual CLIP: xlm-roberta text tower, знает русский
    "xlm_clip_vit_b32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    # SigLIP 2 — мультиязычный (вкл. русский), сильнее базового CLIP
    "siglip2_b16_256":     ("ViT-B-16-SigLIP2-256", "webli"),
    "siglip2_l16_256":     ("ViT-L-16-SigLIP2-256", "webli"),
    "siglip2_so400m_256":  ("ViT-SO400M-16-SigLIP2-256", "webli"),  # ~400M
    "siglip2_gopt_256":    ("ViT-gopt-16-SigLIP2-256", "webli"),    # ~1B
    "siglip2_l16_384":     ("ViT-L-16-SigLIP2-384", "webli"),       # выше разрешение
    # Marqo-FashionSigLIP — доменная (мода), АНГЛИЙСКАЯ (нужен перевод RU→EN)
    "marqo_fashion_siglip": ("hf-hub:Marqo/marqo-fashionSigLIP", ""),
}


def _get_open_clip_name(config_name: str) -> tuple[str, str]:
    """config name -> (open_clip arch, pretrained tag)."""
    if config_name in _MODEL_MAP:
        return _MODEL_MAP[config_name]
    # Позволяем передавать open_clip имя напрямую через config: name=ViT-B-32
    # pretrained задаётся через config.pretrained
    return config_name, "openai"


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------

class _OpenCLIPEncoder(nn.Module):
    """Единый класс для всех open_clip моделей.

    Параметры конфига
    -----------------
    name             : имя из _MODEL_MAP или open_clip arch
    pretrained       : open_clip pretrained tag (override _MODEL_MAP)
    embed_dim        : финальная размерность (None = native бэкбона)
    freeze_backbone  : заморозить визуальный и текстовый трансформеры
    freeze_text      : заморозить только текстовую башню
    freeze_visual    : заморозить только визуальную башню
    grad_checkpointing: экономить GPU-память (медленнее на 10–15%)
    """

    embed_dim: int  # Protocol attribute

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        model_name, default_pretrained = _get_open_clip_name(config["name"])
        pretrained = config.get("pretrained", default_pretrained)
        # hf-hub:* модели (напр. Marqo) несут веса в самом имени → pretrained=None;
        # пустой тег тоже трактуем как None, чтобы не загрузить случайные веса
        if model_name.startswith("hf-hub:") or not pretrained:
            pretrained = None

        log.info("Загружаем open_clip: arch=%s pretrained=%s", model_name, pretrained)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

        # Native embed_dim из бэкбона; config может переопределить для проверки.
        # У TimmModel (SigLIP2) нет .output_dim → пробуем несколько источников,
        # в крайнем случае определяем пробным прогоном картинки.
        native_dim = getattr(model.visual, "output_dim", None)
        if native_dim is None:
            tp = getattr(model, "text_projection", None)
            if isinstance(tp, torch.nn.Parameter):
                native_dim = tp.shape[1]
            elif tp is not None and hasattr(tp, "out_features"):
                native_dim = tp.out_features
        if native_dim is None:
            img_sz = getattr(model.visual, "image_size", 224)
            hw = img_sz if isinstance(img_sz, int) else img_sz[0]
            with torch.no_grad():
                native_dim = model.encode_image(torch.zeros(1, 3, hw, hw)).shape[-1]
        cfg_dim = config.get("embed_dim")
        if cfg_dim and cfg_dim != native_dim:
            log.warning(
                "config.embed_dim=%d != native=%d; используем native=%d",
                cfg_dim, native_dim, native_dim,
            )
        self.embed_dim: int = native_dim

        self._apply_freezing(config)

        if config.get("grad_checkpointing", False):
            try:
                self._model.visual.set_grad_checkpointing(True)
                log.info("Grad checkpointing включён")
            except AttributeError:
                log.warning("Эта архитектура не поддерживает grad checkpointing")

    # -- freezing helpers -----------------------------------------------

    def _freeze_text_tower(self) -> None:
        """Заморозить текстовую башню для обеих архитектур open_clip.

        CustomTextCLIP (напр. xlm-roberta-ViT) держит текст в `.text`;
        обычный CLIP — в `.transformer` + token/positional embeddings, ln_final,
        text_projection. Замораживаем то, что есть.
        """
        text_mod = getattr(self._model, "text", None)
        if text_mod is not None:
            for p in text_mod.parameters():
                p.requires_grad = False
        else:
            for attr in ("transformer", "token_embedding", "ln_final"):
                mod = getattr(self._model, attr, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = False
            for attr in ("positional_embedding", "text_projection"):
                t = getattr(self._model, attr, None)
                if isinstance(t, torch.nn.Parameter):
                    t.requires_grad = False

    def _apply_freezing(self, config: dict[str, Any]) -> None:
        if config.get("freeze_backbone", False):
            for p in self._model.visual.parameters():
                p.requires_grad = False
            self._freeze_text_tower()
            # logit_scale остаётся обучаемым
            log.info("Backbone заморожен (visual + text)")
        else:
            if config.get("freeze_visual", False):
                for p in self._model.visual.parameters():
                    p.requires_grad = False
                log.info("Visual backbone заморожен")
            if config.get("freeze_text", False):
                self._freeze_text_tower()
                log.info("Text tower заморожен")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info("Параметров обучаемых: %d / %d (%.1f%%)", trainable, total, 100 * trainable / max(total, 1))

    # -- Encoder Protocol -----------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) -> (B, embed_dim), L2-norm."""
        emb = self._model.encode_image(images)
        return F.normalize(emb.float(), dim=-1)

    def encode_text(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, L) -> (B, embed_dim), L2-norm.

        attention_mask игнорируется open_clip (он использует маску из EOS-токена),
        но принимается для совместимости с HuggingFace-стилем.
        """
        emb = self._model.encode_text(tokens)
        return F.normalize(emb.float(), dim=-1)

    # -- Удобные методы для скриптов ------------------------------------

    def preprocess_image(self, pil_image):
        """PIL.Image -> Tensor (1, C, H, W), для inference."""
        return self._preprocess(pil_image).unsqueeze(0)

    def tokenize(self, text: str) -> torch.Tensor:
        """str -> Tensor (1, L), для inference."""
        return self._tokenizer([text])

    def get_processor(self):
        """Возвращает (preprocess, tokenizer) как open_clip артефакты."""
        return self._preprocess, self._tokenizer

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход для удобства: -> (img_emb, txt_emb)."""
        return self.encode_image(images), self.encode_text(tokens)


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------

@register("clip_vit_b32")
def _clip_vit_b32(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("clip_vit_b16")
def _clip_vit_b16(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("clip_vit_l14")
def _clip_vit_l14(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("xlm_clip_vit_b32")
def _xlm_clip_vit_b32(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("siglip2_b16_256")
def _siglip2_b16_256(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("siglip2_l16_256")
def _siglip2_l16_256(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("siglip2_so400m_256")
def _siglip2_so400m_256(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("siglip2_gopt_256")
def _siglip2_gopt_256(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("siglip2_l16_384")
def _siglip2_l16_384(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)


@register("marqo_fashion_siglip")
def _marqo_fashion_siglip(config: dict[str, Any]) -> _OpenCLIPEncoder:
    return _OpenCLIPEncoder(config)
