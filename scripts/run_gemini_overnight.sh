#!/bin/bash
# Ночной прогон: ждём 1ч (вдруг придёт 10k Gemini) -> свежий csv -> mix Qwen+Gemini ->
# дообучение L-16 -> eval v1/v2/mix на ДВУХ eval (наш descriptive + Gemini held-out) ->
# выбор лучшей -> ИНДЕКСИРОВАНИЕ каталога лучшей моделью. Утром остаётся только сервер.
set -uo pipefail
cd ~/personal/AAA-project-visual-search
PY=./.venv/bin/python
GPU=2
export SSH_AUTH_SOCK=/tmp/avito_agent.sock
log(){ echo "[$(date '+%F %T')] $*"; }
ev_all(){ grep -E "\[all\]|\[txt\]"; }

log "=== STAGE 0: ожидание ${WAIT_SEC:-0}с ==="
sleep "${WAIT_SEC:-0}"

log "=== STAGE 1: свежий gemini csv ==="
rsync -aq -e ssh "user@[64:ff9b::511a:bcbd]:data/item_queries.csv" data/interim/gemini_item_queries.csv \
  && log "csv обновлён: $(wc -l < data/interim/gemini_item_queries.csv) строк" \
  || log "rsync не удался — использую локальный csv"

log "=== STAGE 2: gemini eval + train + mix ==="
$PY scripts/build_gemini_eval.py 2>&1 | tail -2 || { log "FATAL gemini_eval"; exit 1; }
$PY scripts/build_synth_train.py --llm-queries data/interim/gemini_train_queries.parquet \
  --out data/interim/gemini_train.parquet 2>&1 | tail -1
$PY - <<'EOF'
import pandas as pd
a=pd.read_parquet('data/interim/train_synth_v2.parquet'); b=pd.read_parquet('data/interim/gemini_train.parquet')
pd.concat([a,b[a.columns]],ignore_index=True).to_parquet('data/interim/train_mix.parquet',index=False)
print('train_mix rows:', len(a)+len(b))
EOF

log "=== STAGE 3: fine-tune L-16 на mix ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/finetune_mini.py --model siglip2_l16_256 --device cuda --amp \
  --epochs 2 --batch-size 256 --lr 1e-6 --warmup-frac 0.1 --num-workers 8 --grad-checkpointing \
  --train-parquet data/interim/train_mix.parquet --save-ckpt best \
  --run-name l16_mix --out-dir experiments/l16_mix 2>&1 | grep -E "Epoch|MRR=|img/s" | tail -5

log "=== STAGE 4: eval v1/v2/mix на descriptive + gemini ==="
declare -A DIR=( [v1]=experiments/l16_synth_lr1e-6 [v2]=experiments/l16_synthv2 [mix]=experiments/l16_mix )
for name in v1 v2 mix; do
  ck=$(ls -t ${DIR[$name]}/best_ep*_model_only.pt 2>/dev/null | head -1)
  [ -z "$ck" ] && { log "нет ckpt для $name — пропуск"; continue; }
  log "-- $name descriptive --"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_full.py --model siglip2_l16_256 --ckpt "$ck" \
    --catalog-size 50000 --device cuda --num-workers 8 --out-name ${name}_descr 2>&1 | ev_all
  log "-- $name gemini-heldout --"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_full.py --model siglip2_l16_256 --ckpt "$ck" \
    --catalog-size 50000 --device cuda --num-workers 8 --val-csv data/interim/gemini_eval.csv \
    --out-name ${name}_gemini 2>&1 | ev_all
done

log "=== STAGE 5: выбор лучшей (descr+gemini) ==="
SEL=$($PY - <<'EOF'
import json, glob
last={}
for l in open('experiments/eval_full_ledger.jsonl'):
    r=json.loads(l); last[r['out_name']]=r['metrics']['all']['mrr']
best,bs=None,-1
for n,d in [('v1','experiments/l16_synth_lr1e-6'),('v2','experiments/l16_synthv2'),('mix','experiments/l16_mix')]:
    de=last.get(n+'_descr',0); ge=last.get(n+'_gemini',0); s=de+ge
    print(f'{n}: descr={de:.3f} gemini={ge:.3f} sum={s:.3f}')
    if s>bs: bs,best=s,d
cks=sorted(glob.glob(best+'/best_ep*_model_only.pt'))
print('BEST_CKPT='+(cks[-1] if cks else 'NONE'))
EOF
)
echo "$SEL"
BEST_CKPT=$(echo "$SEL" | sed -n 's/^BEST_CKPT=//p')
log "лучший: $BEST_CKPT"

log "=== STAGE 6: индексирование каталога лучшей моделью ==="
if [ -n "$BEST_CKPT" ] && [ "$BEST_CKPT" != "NONE" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/build_catalog_index.py --model siglip2_l16_256 --ckpt "$BEST_CKPT" \
    --out-index artifacts/catalog.faiss --out-meta artifacts/catalog_meta.parquet --device cuda 2>&1 | tail -4
  cp "$BEST_CKPT" artifacts/model.pt
fi
log "=== DONE ==="
tail -8 experiments/eval_full_ledger.jsonl
log "GEMINI_OVERNIGHT_DONE"
