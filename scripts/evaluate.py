"""CLI: оценка модели на валидационном датасете.

Тонкая обёртка — логика в visual_search.evaluation.evaluate.
Управляется через Hydra-конфиги.

Примеры запуска:
    # Оценка с конфигом по умолчанию
    uv run python scripts/evaluate.py

    # Переопределение путей и параметров через CLI
    uv run python scripts/evaluate.py \
        eval.checkpoint_path=experiments/run_01/checkpoints/best.pt \
        eval.index_path=experiments/run_01/index.faiss \
        eval.output_dir=experiments/run_01/eval_results \
        data.val_path=data/processed/val.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visual_search.common.seed import set_seed
from visual_search.common.logging import get_logger
from visual_search.common.io import EXPERIMENTS_DIR
from visual_search.data.dataset import SearchEvalDataset
from visual_search.models.registry import build_model, get_processor
from visual_search.training.checkpoint import load_checkpoint
from visual_search.evaluation.evaluate import evaluate
from visual_search.index.ann import ANNIndex

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    if config.random_seed.get("fix", True):
        set_seed(config.random_seed.seed, deterministic=config.random_seed.get("deterministic_algorithms", False))

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    logger.info(f"Using device: {device}")

    checkpoint_path = Path(config.eval.checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading model from {checkpoint_path}")
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    model = build_model(model_cfg).to(device)

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    index_path = Path(config.eval.index_path)
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        sys.exit(1)

    logger.info(f"Loading ANN index from {index_path}")
    index = ANNIndex.load(str(index_path))

    logger.info(f"Loading evaluation dataset from {config.data.val_path}")
    processor = get_processor(config.model.pretrained)

    dataset = SearchEvalDataset(
        csv_path=config.data.val_path,
        images_root=config.data.images_root,
        processor=processor,
    )

    logger.info("Starting evaluation...")
    results = evaluate(
        model=model,
        index=index,
        dataset=dataset,
        device=device,
        k_values=config.eval.get("k_values", [1, 5, 10]),
    )

    output_dir = Path(config.eval.get("output_dir", EXPERIMENTS_DIR / "eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    metrics_dict = {mode: metrics.as_flat_dict() for mode, metrics in results.items()}

    out_file = output_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation complete. Metrics saved to {out_file}")


if __name__ == "__main__":
    main()
