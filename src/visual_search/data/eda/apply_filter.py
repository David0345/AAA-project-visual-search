#!/usr/bin/env python3
"""
Two-phase filter for the 1.21M manifest -> relevant.csv

Phase 1 (manifest-only, fast):
  * keep validation_status='ok'
  * cross-item phash dedup: for phash groups with n_items > 1, keep ONE
    representative (priority: is_title=True > min item_id+image_id).
    For phash groups confined to a single item, keep ALL images (different
    angles/lighting of the same item).

Phase 2 (pixel-blank, requires extracted images):
  * drop entropy<1.5 AND edge_density<1.0  (true blank/placeholder)
  * drop aspect<0.3 OR aspect>3.0          (thin collages, bad for image-search)

Each dropped row gets a `drop_reason` column, so we keep the full audit trail
in a single CSV `dropped.csv`.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict


def phase1_manifest(manifest_csv: Path, out_kept: Path, out_dropped: Path) -> dict:
    """Stream the manifest twice: first pass to build phash groups, second to emit."""
    sys.stderr.write(f'Phase 1: reading {manifest_csv}\n')

    # Pass 1: per-phash, count (n_items, n_imgs) + remember candidate rep (is_title preferred)
    # Memory: ~1M phashes * ~80 bytes = ~80 MB
    phash_items: dict[str, set[str]] = defaultdict(set)
    phash_count: dict[str, int] = defaultdict(int)
    phash_rep: dict[str, tuple] = {}   # phash -> (is_title_int, item_id, image_id) used to pick canonical

    n_total = 0
    n_ok = 0
    with manifest_csv.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            n_total += 1
            if row.get('validation_status') != 'ok':
                continue
            n_ok += 1
            ph = row.get('phash') or ''
            if not ph:
                continue
            item_id = row.get('item_id', '')
            image_id = row.get('image_id', '')
            is_title = 1 if (row.get('is_title', '').lower() == 'true') else 0
            phash_items[ph].add(item_id)
            phash_count[ph] += 1
            # higher is_title wins; tie-break by lower item_id then image_id
            key = (-is_title, item_id, image_id)
            cur = phash_rep.get(ph)
            if cur is None or key < cur:
                phash_rep[ph] = key
            if n_total % 200_000 == 0:
                sys.stderr.write(f'  pass1: scanned {n_total:,}\n')

    sys.stderr.write(f'  pass1 done: total={n_total:,}  ok={n_ok:,}  phashes={len(phash_items):,}\n')

    # cross_item phashes -> only one row will be kept (the rep)
    cross_item_phashes = {ph for ph, items in phash_items.items() if len(items) > 1}
    sys.stderr.write(f'  cross-item phash groups: {len(cross_item_phashes):,}\n')

    # Pass 2: write kept / dropped
    n_kept = 0
    n_dropped = defaultdict(int)
    with manifest_csv.open(newline='', encoding='utf-8') as f, \
         out_kept.open('w', newline='', encoding='utf-8') as fk, \
         out_dropped.open('w', newline='', encoding='utf-8') as fd:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        wk = csv.DictWriter(fk, fieldnames=header + ['keep_reason'])
        wk.writeheader()
        wd = csv.DictWriter(fd, fieldnames=header + ['drop_reason'])
        wd.writeheader()

        seen = 0
        for row in r:
            seen += 1
            if seen % 200_000 == 0:
                sys.stderr.write(f'  pass2: scanned {seen:,}\n')

            if row.get('validation_status') != 'ok':
                row['drop_reason'] = 'validation_status_not_ok'
                wd.writerow(row); n_dropped['validation_status_not_ok'] += 1
                continue

            ph = row.get('phash') or ''
            item_id = row.get('item_id', '')
            image_id = row.get('image_id', '')
            is_title = 1 if (row.get('is_title', '').lower() == 'true') else 0

            if ph in cross_item_phashes:
                # cross-item dedup: keep only canonical rep
                rep_key = phash_rep[ph]
                my_key = (-is_title, item_id, image_id)
                if my_key == rep_key:
                    row['keep_reason'] = 'phash_cross_item_rep'
                    wk.writerow(row); n_kept += 1
                else:
                    row['drop_reason'] = 'phash_cross_item_dup'
                    wd.writerow(row); n_dropped['phash_cross_item_dup'] += 1
            else:
                # phash is unique OR appears only within ONE item: keep all
                row['keep_reason'] = 'unique_or_within_item'
                wk.writerow(row); n_kept += 1

    return {
        'total': n_total,
        'ok': n_ok,
        'kept': n_kept,
        'dropped': dict(n_dropped),
        'cross_item_phash_groups': len(cross_item_phashes),
    }


def phase2_pixel(kept_csv: Path, pixel_csv: Path, out_kept: Path, out_dropped: Path,
                 entropy_thr: float = 1.5, edge_thr: float = 1.0,
                 aspect_lo: float = 0.3, aspect_hi: float = 3.0) -> dict:
    """Apply pixel-blank + thin-aspect filter using a pixel_stats CSV
    (image_path/image_id -> entropy, edge_density, aspect).
    """
    # Index pixel stats by logical image_id (the same key as in manifest).
    sys.stderr.write(f'Phase 2: loading pixel stats from {pixel_csv}\n')
    pixel_by_imgid: dict[str, dict] = {}
    with pixel_csv.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('ok') != 'True':
                continue
            iid = row.get('image_id', '')
            if iid:
                pixel_by_imgid[iid] = row
    sys.stderr.write(f'  loaded {len(pixel_by_imgid):,} pixel rows\n')

    n_kept = 0
    n_dropped = defaultdict(int)
    no_pixel = 0
    with kept_csv.open(newline='') as f, \
         out_kept.open('w', newline='') as fk, \
         out_dropped.open('w', newline='') as fd:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        wk = csv.DictWriter(fk, fieldnames=header)
        wk.writeheader()
        wd = csv.DictWriter(fd, fieldnames=header + ['drop_reason_p2'])
        wd.writeheader()

        for row in r:
            iid = row.get('image_id', '')
            pix = pixel_by_imgid.get(iid)
            if pix is None:
                no_pixel += 1
                wk.writerow(row); n_kept += 1
                continue
            try:
                ent = float(pix['entropy'])
                ed = float(pix['edge_density'])
                asp = float(pix['aspect'])
            except Exception:
                wk.writerow(row); n_kept += 1
                continue
            if ent < entropy_thr and ed < edge_thr:
                row['drop_reason_p2'] = f'pixel_blank(ent={ent:.2f},ed={ed:.2f})'
                wd.writerow(row); n_dropped['pixel_blank'] += 1
                continue
            if asp < aspect_lo or asp > aspect_hi:
                row['drop_reason_p2'] = f'pixel_thin(aspect={asp:.2f})'
                wd.writerow(row); n_dropped['pixel_thin'] += 1
                continue
            wk.writerow(row); n_kept += 1

    return {
        'kept': n_kept,
        'dropped': dict(n_dropped),
        'no_pixel_info': no_pixel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/tmp_manifest_validated.csv',
                    help='full validated manifest CSV')
    ap.add_argument('--pixel-stats', default='',
                    help='optional pixel_stats CSV for Phase 2 (skipped if empty)')
    ap.add_argument('--out-dir', default='out')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    p1_kept = out_dir / 'relevant_phase1.csv'
    p1_drop = out_dir / 'dropped_phase1.csv'
    stats1 = phase1_manifest(Path(args.manifest), p1_kept, p1_drop)
    sys.stderr.write(f'Phase 1 stats: {stats1}\n')

    final_kept = p1_kept

    # Phase 2 (only if pixel stats provided)
    if args.pixel_stats:
        p2_kept = out_dir / 'relevant_final.csv'
        p2_drop = out_dir / 'dropped_phase2.csv'
        stats2 = phase2_pixel(p1_kept, Path(args.pixel_stats), p2_kept, p2_drop)
        sys.stderr.write(f'Phase 2 stats: {stats2}\n')
        final_kept = p2_kept

    # write a small summary json
    import json
    summary = {'phase1': stats1}
    if args.pixel_stats:
        summary['phase2'] = stats2
    summary['final_csv'] = str(final_kept)
    (out_dir / 'filter_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.stderr.write(f'final -> {final_kept}\n')


if __name__ == '__main__':
    main()
