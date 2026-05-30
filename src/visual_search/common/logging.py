"""Единая настройка логирования для всех точек входа.

TODO(common): get_logger(name) с общим форматом; уровень из env/конфига.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
