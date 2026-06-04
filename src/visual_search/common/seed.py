"""Фиксация сидов для воспроизводимости прогонов.

Один и тот же конфиг + тот же сид -> тот же результат (см. §7 PROJECT_STRUCTURE).

TODO(common): seed_everything(seed) — random, numpy, torch, cudnn.
"""

from __future__ import annotations
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


set_seed = seed_everything
