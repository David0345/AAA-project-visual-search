"""Реализации энкодеров — обёртки над open_clip / transformers (CLIP, SigLIP).

Каждая реализация удовлетворяет контракту base.Encoder и регистрируется в
registry. Здесь же живёт логика заморозки/разморозки слоёв при дотюнинге.

TODO(Обучение): первая обёртка (например, open_clip ViT-B/32) + @register.
"""

from __future__ import annotations
