"""CLI: запуск обучения.

Тонкая обёртка — логика в visual_search.training.train.run.
Запуск: python scripts/train.py --config configs/experiment/clip_baseline.yaml
"""

from __future__ import annotations
import hydra
from omegaconf import DictConfig
import torch

from visual_search.training.train import run
from visual_search.common.seed import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:

    if config.random_seed.fix:
        set_seed(config.random_seed.seed, deterministic=config.random_seed.deterministic_algorithms)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    run(config, device)


if __name__ == "__main__":
    main()
