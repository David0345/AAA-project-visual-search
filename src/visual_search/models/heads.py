"""Проекционные головы.
Для RuCLIP (CLIPModel из transformers) проекционные головы уже встроены
в саму модель (visual_projection и text_projection) и применяются внутри
get_image_features / get_text_features.
Поэтому отдельный модуль для голов не требуется.
"""

from __future__ import annotations
