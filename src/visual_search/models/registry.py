"""Фабрика моделей и процессоров: «имя из конфига -> экземпляр».

Добавить модель = написать класс в encoders.py и повесить @register("имя").
training/evaluation/serving зовут только build_model/get_processor.

Пример:
    from visual_search.models.registry import build_model, get_processor
    model = build_model({"name": "clip_vit_b32", "pretrained": "openai"})
    preprocess, tokenizer = get_processor("openai")
"""

from __future__ import annotations

import importlib
from typing import Any

from visual_search.models.base import Encoder

# name -> callable(config) -> Encoder
_REGISTRY: dict[str, Any] = {}


def register(name: str):
    """Декоратор для регистрации реализации энкодера под именем."""
    def _wrap(factory):
        _REGISTRY[name] = factory
        return factory
    return _wrap


def _ensure_encoders_imported() -> None:
    """Lazy-импорт encoders.py, чтобы все @register выполнились."""
    if not _REGISTRY:
        importlib.import_module("visual_search.models.encoders")


def build_model(config: dict[str, Any]) -> Encoder:
    """Создать энкодер по config['name'].

    Args:
        config: dict с обязательным ключом 'name' и опциональными параметрами
                (pretrained, embed_dim, freeze_backbone, ...).

    Returns:
        Экземпляр, реализующий протокол Encoder.
    """
    _ensure_encoders_imported()
    name = config["name"]
    if name not in _REGISTRY:
        raise KeyError(
            f"Неизвестная модель {name!r}. "
            f"Зарегистрированы: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](config)


def get_processor(pretrained: str = "openai") -> "_CLIPProcessorAdapter":
    """Вернуть процессор (preprocess + tokenizer) для заданного pretrained.

    Используется в datamodule.py и scripts/evaluate.py.

    Args:
        pretrained: тег весов (openai | laion5b_s13b_b90k | ...).

    Returns:
        _CLIPProcessorAdapter — совместим с CLIPProcessor-интерфейсом dataset.py.
    """
    import open_clip

    # Архитектура по тегу (все наши модели — ViT-B-32)
    arch_map = {
        "openai":             "ViT-B-32",
        "laion5b_s13b_b90k":  "xlm-roberta-base-ViT-B-32",
    }
    arch = arch_map.get(pretrained, "ViT-B-32")

    _, preprocess, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    return _CLIPProcessorAdapter(preprocess, tokenizer)


class _CLIPProcessorAdapter:
    """Адаптер open_clip preprocess + tokenizer под интерфейс dataset.py.

    dataset.py ожидает объект, работающий как HuggingFace CLIPProcessor:
        processor(text=[...], images=[...], return_tensors="pt", ...) -> dict
    """

    def __init__(self, preprocess, tokenizer) -> None:
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    def __call__(
        self,
        text: list[str] | None = None,
        images=None,
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
        **_,
    ) -> dict:
        import torch

        result: dict = {}

        if images is not None:
            pixel_values = torch.stack([self._preprocess(img) for img in images])
            result["pixel_values"] = pixel_values

        if text is not None:
            tokens = self._tokenizer(text)  # (N, L)
            result["input_ids"] = tokens
            result["attention_mask"] = (tokens != 0).long()

        return result

    @property
    def preprocess(self):
        return self._preprocess

    @property
    def tokenizer(self):
        return self._tokenizer
