#!/bin/bash
# Ночной конвейер: дождаться rsync картинок -> собрать train_full.parquet -> обучить.
# Запускается в tmux (сессия avito_train), лог: experiments/full_pipeline.log
set -uo pipefail
cd ~/personal/AAA-project-visual-search
PY=./.venv/bin/python
export SSH_AUTH_SOCK=/tmp/avito_agent.sock
GPU=1
RSYNC_PID=${1:-1381009}

log() { echo "[$(date '+%F %T')] $*"; }

log "=== STAGE 1: ждём завершения rsync (pid $RSYNC_PID) ==="
while kill -0 "$RSYNC_PID" 2>/dev/null; do sleep 60; done
N_JPG=$(find data/raw/dataset_1M/images -name '*.jpg' | wc -l)
log "rsync завершился. Локально jpg: $N_JPG"

if [ "$N_JPG" -lt 300000 ]; then
  log "Недокачано (<300k) — повторный rsync (докачает только недостающее)"
  rsync -a --info=stats2 --files-from=data/interim/need_title_images.txt \
    "user@[64:ff9b::511a:bcbd]:data/upload/dataset_1M/" data/raw/dataset_1M/
  N_JPG=$(find data/raw/dataset_1M/images -name '*.jpg' | wc -l)
  log "После повтора jpg: $N_JPG (учим на том, что есть — prepare сам отфильтрует)"
fi

log "=== STAGE 2: подмена CSV на полные + сборка train_full.parquet ==="
cd data/raw/dataset_1M
[ -L images.csv ] || { mv images.csv images.sample50k.csv && ln -s ~/personal/images.csv images.csv; }
[ -L tmp_manifest_with_urls.csv ] || { mv tmp_manifest_with_urls.csv tmp_manifest_with_urls.sample50k.csv \
  && ln -s ~/personal/tmp_manifest_with_urls.csv tmp_manifest_with_urls.csv; }
cd ~/personal/AAA-project-visual-search

$PY scripts/prepare_mini_train.py \
  --data-dir data/raw/dataset_1M \
  --valid-ids src/visual_search/data/eda/valid_image_ids.csv \
  --output data/interim/train_full.parquet || { log "FATAL: prepare упал"; exit 1; }

log "=== STAGE 3: обучение winner-рецептом (lr=2e-6, bs512, 3 эпохи, eval каждую) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --device cuda --amp \
  --epochs 3 --batch-size 512 --lr 2e-6 --warmup-frac 0.1 --num-workers 8 \
  --train-parquet data/interim/train_full.parquet \
  --save-ckpt best --run-name full_lr2e-6_bs512 --out-dir experiments/full_lr2e-6

log "=== STAGE 4: ночной мини-свип lr на полном сете (без чекпойнтов) ==="
for LR in 1e-6 5e-6; do
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --device cuda --amp \
    --epochs 2 --batch-size 512 --lr $LR --warmup-frac 0.1 --num-workers 8 \
    --train-parquet data/interim/train_full.parquet \
    --save-ckpt none --run-name full_lr${LR}_bs512 --out-dir experiments/full_lr${LR}
done

log "=== DONE. Итоговая таблица: ==="
$PY scripts/show_results.py
