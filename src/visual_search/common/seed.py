"""Фиксация сидов для воспроизводимости прогонов.

Один и тот же конфиг + тот же сид -> тот же результат (см. §7 PROJECT_STRUCTURE).

TODO(common): seed_everything(seed) — random, numpy, torch, cudnn.
"""

from __future__ import annotations


def seed_everything(seed: int) -> None:
    raise NotImplementedError
