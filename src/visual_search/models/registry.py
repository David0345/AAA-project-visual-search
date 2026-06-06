"""Фабрика моделей и процессоров: «имя из конфига -> экземпляр Encoder».
Точка расширяемости. training/evaluation/serving зовут только
build_model(config) и get_processor(name) и не знают про конкретные классы.
"""

from __future__ import annotations

from typing import Any
from transformers import CLIPProcessor

from visual_search.models.base import Encoder

# name -> callable(config) -> Encoder
_REGISTRY: dict[str, Any] = {}


def register(name: str):
    """Декоратор для регистрации реализации энкодера под именем."""
    def _wrap(factory):
        _REGISTRY[name] = factory
        return factory
    return _wrap


def build_model(config: dict[str, Any]) -> Encoder:
    """Создать энкодер по config['name']."""
    name = config["name"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config)


def get_processor(model_name: str):
    """
    Загрузить процессор (токенизатор + препроцессинг изображений) для модели.
    model_name здесь — это HF id.
    Используется в datamodule.py и scripts/evaluate.py.
    """
    return CLIPProcessor.from_pretrained(model_name)
