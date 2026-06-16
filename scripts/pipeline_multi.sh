#!/bin/bash
# Ждёт докачку доп.ракурсов -> собирает train_full_multi.parquet -> обучает
# winner-рецептом с мульти-фото ресэмплингом. Сравнение с single-image winner.
set -uo pipefail
cd ~/personal/AAA-project-visual-search
PY=./.venv/bin/python
GPU=1
log() { echo "[$(date '+%F %T')] $*"; }

log "=== STAGE 1: ждём завершения rsync доп.ракурсов ==="
# rsync процесс узнаём по аргументу need_extra_images.txt
while pgrep -f "need_extra_images.txt" >/dev/null 2>&1; do sleep 60; done
N=$(find data/raw/dataset_1M/images -name '*.jpg' | wc -l)
log "rsync завершён. Всего локальных jpg: $N"

log "=== STAGE 2: сборка train_full_multi.parquet ==="
$PY scripts/build_multi_parquet.py || { log "FATAL: build упал"; exit 1; }

log "=== STAGE 3: обучение winner-рецептом + мульти-фото (lr=5e-6, bs512, 3 эпохи) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --device cuda --amp \
  --epochs 3 --batch-size 512 --lr 5e-6 --warmup-frac 0.1 --num-workers 8 \
  --train-parquet data/interim/train_full_multi.parquet \
  --save-ckpt best --run-name multi_lr5e-6_bs512 --out-dir experiments/multi_lr5e-6

log "=== DONE. Таблица: ==="
$PY scripts/show_results.py
