"""Оркестрация прогона: конфиг -> данные + модель + loop -> чекпойнт.

Это логика; запускается тонкой обёрткой scripts/train.py.

TODO(Обучение): run(config) — seed, datamodule, build_model, optim, loop, save.
"""

from __future__ import annotations

from typing import Any


def run(config: dict[str, Any]) -> None:
    raise NotImplementedError
