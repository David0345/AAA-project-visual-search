#!/usr/bin/env python3
"""
Compute pixel_stats for every kept row in relevant_phase1.csv.
Resolves image_path against the locally-extracted dataset root and reuses
_pixel_stats_for_path from eda.py (entropy, edge_density, aspect, ...).

Output CSV columns match what apply_filter.py Phase 2 expects:
  image_id, item_id, path, predmet_odezhdy, size_bytes, width, height, mode,
  aspect, mean_brightness, std_brightness, entropy, edge_density,
  mean_R, mean_G, mean_B, channel_spread, redness, ok, error
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Reuse the worker function and path resolver from eda.py
sys.path.insert(0, str(Path(__file__).parent))
from eda import _pixel_stats_for_path, _resolve_image_path  # noqa: E402

FIELDS = [
    "image_id", "item_id", "path", "predmet_odezhdy",
    "size_bytes", "width", "height", "mode", "aspect",
    "mean_brightness", "std_brightness", "entropy", "edge_density",
    "mean_R", "mean_G", "mean_B", "channel_spread", "redness",
    "ok", "error",
]


def iter_args(kept_csv: Path, images_root: Path):
    with kept_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            image_path_in_csv = row.get("image_path", "")
            if not image_path_in_csv:
                continue
            p = _resolve_image_path(image_path_in_csv, images_root)
            if not p.exists():
                continue
            try:
                item_id = int(row["item_id"]) if row.get("item_id") else None
            except ValueError:
                item_id = None
            try:
                image_id = int(row["image_id"]) if row.get("image_id") else None
            except ValueError:
                image_id = None
            yield (str(p), item_id, image_id, row.get("predmet_odezhdy", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept-csv", default="out/relevant_phase1.csv")
    ap.add_argument("--images-root", default="data/full/dataset_1M/images",
                    help="root under which images/NNN/MMM/x.jpg live")
    ap.add_argument("--out-csv", default="out/pixel_stats_full.csv")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    kept = Path(args.kept_csv)
    images_root = Path(args.images_root)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    sys.stderr.write(f"reading kept rows from {kept}\n")
    sys.stderr.write(f"images_root = {images_root}\n")
    sys.stderr.write(f"workers = {args.workers}, chunksize = {args.chunksize}\n")

    # Stream the input -> stream the output. Don't materialize 1M args.
    t0 = time.time()
    n_in = 0
    n_ok = 0
    n_fail = 0
    last_log = t0

    arg_gen = iter_args(kept, images_root)
    if args.limit > 0:
        def _capped(it, n):
            for i, v in enumerate(it):
                if i >= n:
                    return
                yield v
        arg_gen = _capped(arg_gen, args.limit)

    with out.open("w", newline="", encoding="utf-8") as fo, \
         ProcessPoolExecutor(max_workers=args.workers) as ex:
        w = csv.DictWriter(fo, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for result in ex.map(_pixel_stats_for_path, arg_gen, chunksize=args.chunksize):
            n_in += 1
            if result.get("ok"):
                n_ok += 1
            else:
                n_fail += 1
            w.writerow(result)
            now = time.time()
            if now - last_log >= 10:
                rate = n_in / (now - t0)
                sys.stderr.write(
                    f"  processed={n_in:,}  ok={n_ok:,}  failed={n_fail:,}  "
                    f"rate={rate:.0f} img/s\n"
                )
                last_log = now

    dt = time.time() - t0
    sys.stderr.write(
        f"done: processed={n_in:,}  ok={n_ok:,}  failed={n_fail:,}  "
        f"elapsed={dt:.1f}s ({n_in/max(dt,1e-3):.0f} img/s)\n"
    )
    sys.stderr.write(f"-> {out}\n")


if __name__ == "__main__":
    main()
