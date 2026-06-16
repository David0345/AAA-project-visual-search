#!/bin/bash
# Resume: генерация уже сделана (queries_qwen.parquet). Делаем build + fine-tune + eval.
set -uo pipefail
cd ~/personal/AAA-project-visual-search
PY=./.venv/bin/python
GPU=1
log() { echo "[$(date '+%F %T')] $*"; }

eval_ckpt() {  # $1=model $2=run_dir $3=out_name
  local ckpt; ckpt=$(ls -t "$2"/best_ep*_model_only.pt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then log "НЕТ чекпойнта в $2"; return 1; fi
  log "eval_full $3 (ckpt=$ckpt)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_full.py --model "$1" --ckpt "$ckpt" \
    --catalog-size 50000 --device cuda --num-workers 8 --out-name "$3" 2>&1 | grep -E "\[image\]|\[txt\]|\[multimodal\]|\[all\]"
}

log "=== STAGE 2: build train_synth ==="
$PY scripts/build_synth_train.py --llm-queries data/interim/queries_qwen.parquet \
  --out data/interim/train_synth.parquet 2>&1 | tail -3 || { log "FATAL build"; exit 1; }

log "=== STAGE 3: fine-tune L-16 (свип lr) ==="
for LR in 1e-6 2e-6; do
  log "--- L-16 synth lr=$LR ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --model siglip2_l16_256 --device cuda --amp \
    --epochs 2 --batch-size 256 --lr $LR --warmup-frac 0.1 --num-workers 8 --grad-checkpointing \
    --train-parquet data/interim/train_synth.parquet --save-ckpt best \
    --run-name l16_synth_lr${LR} --out-dir experiments/l16_synth_lr${LR} 2>&1 | grep -E "Epoch|MRR=|img/s" | tail -6
  eval_ckpt siglip2_l16_256 experiments/l16_synth_lr${LR} l16_synth_lr${LR}_50k
done

log "=== STAGE 4: fine-tune gopt lr=1e-6 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --model siglip2_gopt_256 --device cuda --amp \
  --epochs 2 --batch-size 128 --lr 1e-6 --warmup-frac 0.1 --num-workers 8 --grad-checkpointing \
  --train-parquet data/interim/train_synth.parquet --save-ckpt best \
  --run-name gopt_synth_lr1e-6 --out-dir experiments/gopt_synth_lr1e-6 2>&1 | grep -E "Epoch|MRR=|img/s|CUDA out|Error" | tail -6
eval_ckpt siglip2_gopt_256 experiments/gopt_synth_lr1e-6 gopt_synth_lr1e-6_50k

log "=== DONE ==="
tail -6 experiments/eval_full_ledger.jsonl
log "SYNTH_RESUME_DONE"
