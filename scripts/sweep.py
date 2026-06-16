#!/usr/bin/env python3
"""Стейджированный свип гиперпараметров поверх scripts/finetune_mini.py.

Каждый прогон запускается отдельным процессом (чистая CUDA-память, изоляция OOM)
с --save-ckpt none — чекпойнты НЕ пишутся (общий диск). Все прогоны сами
дописываются в experiments/metrics_ledger.jsonl. В конце — таблица show_results.py.

Запуск:
    .venv/bin/python scripts/sweep.py --gpu 1
    .venv/bin/python scripts/sweep.py --gpu 1 --only A B   # только стадии A,B
    .venv/bin/python scripts/sweep.py --gpu 1 --dry-run    # показать матрицу
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
FINETUNE = ROOT / "scripts" / "finetune_mini.py"
PY = str(ROOT / ".venv" / "bin" / "python")

# База, общая для всех прогонов свипа (eval каждую эпоху, чекпойнты не пишем).
BASE = dict(epochs=2, num_workers=8, warmup_frac=0.1, loss="infonce")


def _runs() -> list[tuple[str, dict]]:
    """(stage, config) — config переопределяет BASE. run_name генерится из стадии."""
    runs: list[tuple[str, dict]] = []

    # --- A. LR (bs512, infonce, warmup0.1) ---
    for lr in (2e-6, 5e-6, 1e-5, 2e-5):
        runs.append(("A", dict(batch_size=512, lr=lr,
                               run_name=f"A_lr{lr:g}_bs512")))

    # --- B. Batch size (lr=1e-5, infonce, warmup0.1) ---
    runs.append(("B", dict(batch_size=256, lr=1e-5, run_name="B_bs256_lr1e-5")))
    runs.append(("B", dict(batch_size=1024, lr=1e-5, run_name="B_bs1024_lr1e-5")))
    runs.append(("B", dict(batch_size=2048, lr=1e-5, grad_checkpointing=True,
                           run_name="B_bs2048_lr1e-5_gckpt")))

    # --- C. Архитектура (bs512, lr=1e-5, warmup0.1) ---
    runs.append(("C", dict(batch_size=512, lr=1e-5, freeze_text=True,
                           run_name="C_freezetext_bs512_lr1e-5")))
    runs.append(("C", dict(batch_size=512, lr=1e-5, freeze_visual=True,
                           run_name="C_freezevisual_bs512_lr1e-5")))
    runs.append(("C", dict(batch_size=512, lr=1e-5, loss="sigmoid",
                           run_name="C_sigmoid_bs512_lr1e-5")))
    runs.append(("C", dict(batch_size=512, lr=1e-5, warmup_frac=0.0,
                           run_name="C_nowarmup_bs512_lr1e-5")))
    return runs


def _build_cmd(cfg: dict, gpu: int) -> list[str]:
    merged = {**BASE, **cfg}
    cmd = [PY, str(FINETUNE), "--device", "cuda", "--amp", "--save-ckpt", "none"]
    out_dir = ROOT / "experiments" / "sweep" / merged["run_name"]
    cmd += ["--out-dir", str(out_dir)]
    for k, v in merged.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)       # store_true
        else:
            cmd += [flag, str(v)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--only", nargs="+", default=None, help="Стадии: A B C")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = _runs()
    if args.only:
        runs = [r for r in runs if r[0] in set(args.only)]

    print(f"Свип: {len(runs)} прогонов на GPU {args.gpu}\n")
    for stage, cfg in runs:
        print(f"  [{stage}] {cfg['run_name']}: " +
              " ".join(_build_cmd(cfg, args.gpu)[3:]))
    if args.dry_run:
        return

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    t0 = time.time()
    ok, failed = [], []
    for i, (stage, cfg) in enumerate(runs, 1):
        name = cfg["run_name"]
        print(f"\n{'='*70}\n[{i}/{len(runs)}] [{stage}] {name}\n{'='*70}", flush=True)
        cmd = _build_cmd(cfg, args.gpu)
        try:
            r = subprocess.run(cmd, env=env, cwd=str(ROOT))
            (ok if r.returncode == 0 else failed).append(name)
            if r.returncode != 0:
                print(f"!! {name} упал (returncode={r.returncode}) — продолжаем", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"!! {name} исключение: {e} — продолжаем", flush=True)

    dt = (time.time() - t0) / 60
    print(f"\n{'='*70}\nСвип готов за {dt:.1f} мин. ok={len(ok)} failed={len(failed)}")
    if failed:
        print("Упавшие:", ", ".join(failed))

    print("\n--- Результаты ---")
    subprocess.run([PY, str(ROOT / "scripts" / "show_results.py")], cwd=str(ROOT))


if __name__ == "__main__":
    main()
