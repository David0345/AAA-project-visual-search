#!/bin/bash
# v2 синтетика: 5-стилевые запросы (короткий/стандартный/детальный/разговорный/атрибутивный)
# параллельно на 3 GPU -> merge -> train_synth_v2 -> fine-tune L-16 + eval_full.
# Стили закрывают спор «микс/замена»: атрибутивный = старые шаблоны, короткий/разговорный = юзер-стиль.
set -uo pipefail
cd ~/personal/AAA-project-visual-search
PY=./.venv/bin/python
TOTAL=90000
GPUS=(2 3 4)
NSHARD=${#GPUS[@]}
MAXWAIT=39600   # 11h защитный лимит
log(){ echo "[$(date '+%F %T')] $*"; }

log "=== STAGE 1: генерация $TOTAL товаров на $NSHARD GPU (${GPUS[*]}) ==="
rm -f data/interim/qwen_v2_shard*.parquet data/interim/qwen_v2_shard*.parquet.done
for i in $(seq 0 $((NSHARD-1))); do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g $PY scripts/gen_queries_qwen.py --full \
    --total $TOTAL --num-shards $NSHARD --shard-id $i --batch-size 8 \
    --out data/interim/qwen_v2_shard${i}.parquet > experiments/gen_v2_shard${i}.log 2>&1 &
done

log "ждём $NSHARD шардов (лимит ${MAXWAIT}s) ..."
waited=0
while [ "$(ls data/interim/qwen_v2_shard*.parquet.done 2>/dev/null | wc -l)" -lt "$NSHARD" ]; do
  sleep 60; waited=$((waited+60))
  if [ "$waited" -ge "$MAXWAIT" ]; then log "ТАЙМАУТ — продолжаю с готовыми шардами"; break; fi
done
ndone=$(ls data/interim/qwen_v2_shard*.parquet.done 2>/dev/null | wc -l)
log "готово шардов: $ndone / $NSHARD"

log "=== STAGE 2: merge + build_synth ==="
$PY - <<'EOF'
import pandas as pd, glob
parts=[pd.read_parquet(p) for p in sorted(glob.glob('data/interim/qwen_v2_shard*.parquet'))]
df=pd.concat(parts, ignore_index=True).drop_duplicates('item_id')
df.to_parquet('data/interim/queries_qwen_v2.parquet', index=False)
print('merged items:', len(df))
EOF
$PY scripts/build_synth_train.py --llm-queries data/interim/queries_qwen_v2.parquet \
  --out data/interim/train_synth_v2.parquet 2>&1 | tail -3 || { log "FATAL build"; exit 1; }

log "=== STAGE 3: fine-tune L-16 на synth_v2 + eval ==="
CUDA_VISIBLE_DEVICES=2 $PY scripts/finetune_mini.py --model siglip2_l16_256 --device cuda --amp \
  --epochs 2 --batch-size 256 --lr 1e-6 --warmup-frac 0.1 --num-workers 8 --grad-checkpointing \
  --train-parquet data/interim/train_synth_v2.parquet --save-ckpt best \
  --run-name l16_synthv2 --out-dir experiments/l16_synthv2 2>&1 | grep -E "Epoch|MRR=|img/s|CUDA out|Error" | tail -6
ckpt=$(ls -t experiments/l16_synthv2/best_ep*_model_only.pt 2>/dev/null | head -1)
if [ -n "$ckpt" ]; then
  CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_full.py --model siglip2_l16_256 --ckpt "$ckpt" \
    --catalog-size 50000 --device cuda --num-workers 8 --out-name l16_synthv2_50k 2>&1 | grep -E "\[image\]|\[txt\]|\[multimodal\]|\[all\]"
fi
log "=== DONE ==="
tail -4 experiments/eval_full_ledger.jsonl
log "SYNTHV2_DONE"
