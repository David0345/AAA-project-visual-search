"""models — определения энкодеров, головы, лоссы и фабрика моделей.

ВЛАДЕЛЕЦ: «Обучение».

Внешний код (training/, evaluation/, serving/) НЕ импортирует конкретные классы
отсюда напрямую — только через registry.build_model(config). Это и есть точка
гибкости: новый бэкбон добавляется здесь, остальной код не меняется.
"""

from visual_search.models.base import Encoder
from visual_search.models.registry import build_model, register, get_processor
from visual_search.models.encoders import RuCLIPEncoder

__all__ = ["Encoder", "build_model", "register", "get_processor", "RuCLIPEncoder"]
