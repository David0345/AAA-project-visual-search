"""Загрузка и валидация конфигов экспериментов.

Конфиг — композиция из configs/{data,model,train}/ (см. configs/experiment/).
Схема конфига живёт ЗДЕСЬ, в одном месте: добавили поле в YAML -> добавили его
в валидацию здесь. Тогда все модули видят согласованный набор параметров.

TODO(common): реализовать load_config (yaml + слияние ссылок на под-конфиги).
Опционально позже — перейти на OmegaConf/Hydra для override из CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Прочитать experiment-конфиг и собрать его из под-конфигов.

    Возвращает провалидированный словарь с секциями data/model/train.
    """
    raise NotImplementedError
