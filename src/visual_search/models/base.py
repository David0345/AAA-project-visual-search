"""Контракт энкодера — ядро всей системы.

Любая модель реализует этот интерфейс. На него опираются обучение, оценка и
сервис. Пока контракт держится, реализацию можно менять свободно.

Инварианты:
  * эмбеддинги L2-нормированы -> схожесть = косинус = скалярное произведение;
  * embed_dim объявлен и неизменен в рамках одного индекса;
  * encode_image / encode_text возвращают тензор (B, embed_dim).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from torch import Tensor


@runtime_checkable
class Encoder(Protocol):
    """Единый интерфейс двухбашенного мультимодального энкодера."""
    embed_dim: int

    def encode_image(self, images: Tensor) -> Tensor:
        """images: (B, C, H, W) -> (B, embed_dim), L2-norm."""
        ...

    def encode_text(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        tokens: (B, L)
        attention_mask: (B, L), optional. Игнорирует паддинг-токены при вычислении эмбеддинга.
        -> (B, embed_dim), L2-norm.
        """
        ...
