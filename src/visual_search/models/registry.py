"""Фабрика моделей: «имя из конфига -> экземпляр Encoder».

Точка расширяемости. Добавить модель = написать класс в encoders.py и
зарегистрировать его здесь. training/evaluation/serving зовут только
build_model(config) и не знают про конкретные классы.

TODO(Обучение): реализовать реестр и build_model.

Пример целевого использования:
    from visual_search.models import build_model
    model = build_model({"name": "clip_vit_b32", "embed_dim": 512, ...})
"""

from __future__ import annotations

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


def build_model(config: dict[str, Any]) -> Encoder:
    """Создать энкодер по config['name']."""
    name = config["name"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config)
