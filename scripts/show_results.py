#!/usr/bin/env python3
"""Рендер истории замеров (experiments/metrics_ledger.jsonl) в markdown-таблицу.

Источник истины — append-only ledger, который пишет scripts/finetune_mini.py.
Сверху печатается zero-shot baseline из experiments/zeroshot/<model>/metrics.json,
чтобы сразу видеть, не деградирует ли txt после дообучения.

Запуск:
    .venv/bin/python scripts/show_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import EXPERIMENTS_DIR


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    ledger = EXPERIMENTS_DIR / "metrics_ledger.jsonl"

    # --- Zero-shot baseline ---
    print("# Результаты\n")
    zs_path = EXPERIMENTS_DIR / "zeroshot" / "xlm_clip_vit_b32" / "metrics.json"
    if zs_path.exists():
        zs = json.loads(zs_path.read_text())
        print("## Zero-shot baseline (xlm_clip_vit_b32, каталог=477)\n")
        print("| mode | MRR | R@10 |")
        print("|---|---|---|")
        for m in ("image", "txt", "multimodal", "all"):
            if m in zs:
                print(f"| {m} | {_fmt(zs[m].get('mrr'))} | {_fmt(zs[m].get('recall@10'))} |")
        print()

    # --- Прогоны обучения ---
    if not ledger.exists():
        print("_(ledger пуст — ещё не было прогонов с eval)_")
        return

    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    base_txt = 0.602  # zero-shot txt MRR (anchor)
    if zs_path.exists():
        base_txt = json.loads(zs_path.read_text()).get("txt", {}).get("mrr", 0.602)

    def _at_best(r, key):
        """Значение метрики на лучшей эпохе (по all MRR); fallback — последняя."""
        traj = r.get(f"{key}_mrr_trajectory") or []
        be = r.get("best_epoch")
        if be and 1 <= be <= len(traj) and traj[be - 1] is not None:
            return traj[be - 1]
        return (r.get("eval") or {}).get(key, {}).get("mrr")

    print("## Дообучение — метрики на ЛУЧШЕЙ эпохе (txt не должен падать; mm/all — рост)\n")
    print("| run | bs | lr | loss | freeze | warm | img/s | best ep | txt MRR | Δtxt | mm MRR | all MRR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        txt = _at_best(r, "txt")
        mm = _at_best(r, "multimodal")
        al = _at_best(r, "all")
        delta = (txt - base_txt) if txt is not None else None
        dstr = (f"+{delta:.3f}" if (delta is not None and delta >= 0)
                else (f"{delta:.3f}" if delta is not None else "—"))
        print(f"| {r.get('run_name')} | {r.get('batch_size')} | {r.get('lr')} | "
              f"{r.get('loss','—')} | {r.get('freeze') or '—'} | {r.get('warmup_frac','—')} | "
              f"{r.get('imgs_per_sec')} | {r.get('best_epoch') or '—'} | "
              f"{_fmt(txt)} | {dstr} | {_fmt(mm)} | {_fmt(al)} |")


if __name__ == "__main__":
    main()
