#!/bin/bash
# Свип lr на полном сете (226 979 товаров) вокруг нового оптимума (5e-6 был лучшим и ещё рос).
# 3 эпохи, eval каждую, сохраняем лучший чекпойнт каждого прогона.
set -uo pipefail
cd ~/personal/AAA-project-visual-search
PY=./.venv/bin/python
GPU=1
log() { echo "[$(date '+%F %T')] $*"; }

for LR in 5e-6 7e-6 1e-5; do
  log "=== full sweep lr=$LR (3 эпохи) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --device cuda --amp \
    --epochs 3 --batch-size 512 --lr $LR --warmup-frac 0.1 --num-workers 8 \
    --train-parquet data/interim/train_full.parquet \
    --save-ckpt best --run-name fullsweep_lr${LR}_bs512 --out-dir experiments/fullsweep_lr${LR}
done

log "=== DONE. Таблица: ==="
$PY scripts/show_results.py
