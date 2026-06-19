"""models — определения энкодеров, головы, лоссы и фабрика моделей.

Внешний код (training/, evaluation/, serving/) НЕ импортирует конкретные классы
отсюда напрямую — только через registry.build_model(config).
"""

from visual_search.models.base import Encoder
from visual_search.models.registry import build_model

__all__ = ["Encoder", "build_model"]
