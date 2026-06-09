#!/usr/bin/env python3
"""Smoke-тест файн-тюнинга xlm_clip_vit_b32 на mini_train.parquet (MPS/CPU).

Цель: оценить потенциальный прирост метрик и скорость обучения
      перед запуском полноценного обучения на GPU.

Запуск:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/finetune_mini.py
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/finetune_mini.py --epochs 2 --batch-size 24
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/finetune_mini.py --device cpu --epochs 1

После каждой эпохи сравниваем с zero-shot baseline (xlm MRR=0.602 на txt).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Zero-shot baseline для сравнения (из experiments/zeroshot/xlm_clip_vit_b32/)
BASELINE = {
    "txt":        {"mrr": 0.602, "recall@10": 0.282},
    "image":      {"mrr": 0.001, "recall@10": 0.001},
    "multimodal": {"mrr": 0.062, "recall@10": 0.072},
    "all":        {"mrr": 0.227, "recall@10": 0.121},
}


# ---------------------------------------------------------------------------
# Dataset — тонкая обёртка над ContrastiveImageTextDataset
# ---------------------------------------------------------------------------

def build_train_loader(
    parquet_path: str,
    images_root: str,
    preprocess,
    tokenizer,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
) -> DataLoader:
    from torch.utils.data import Dataset

    import pandas as pd
    from PIL import Image
    import ast

    class _MiniDataset(Dataset):
        def __init__(self, parquet_path, images_root, preprocess, tokenizer, seed):
            self.df = pd.read_parquet(parquet_path)
            self.images_root = Path(images_root)
            self.preprocess = preprocess
            self.tokenizer = tokenizer
            self.rng = np.random.default_rng(seed)

            # Парсим queries из строки если нужно
            if "queries" in self.df.columns and self.df["queries"].dtype == object:
                self.df["queries"] = self.df["queries"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

            # Убираем строки с пустыми queries
            self.df = self.df[self.df["queries"].map(len) > 0].reset_index(drop=True)
            log.info("Dataset: %d items", len(self.df))

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            # Изображение
            img_path = self.images_root / row["title_image_path"]
            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(image)
            except Exception as e:
                log.debug("Ошибка загрузки %s: %s", img_path, e)
                img_tensor = torch.zeros(3, 224, 224)

            # Текст: случайный запрос
            queries = row["queries"]
            q_idx = int(self.rng.integers(len(queries)))
            text = queries[q_idx]
            tokens = self.tokenizer([text])  # (1, L)

            return img_tensor, tokens.squeeze(0), text

    def collate(batch):
        imgs, tokens, texts = zip(*batch)
        return (
            torch.stack(imgs),          # (B, C, H, W)
            torch.stack(tokens),        # (B, L)
            list(texts),
        )

    ds = _MiniDataset(parquet_path, images_root, preprocess, tokenizer, seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS не поддерживает pin_memory
        collate_fn=collate,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Evaluation (переиспользуем логику из zeroshot_eval.py)
# ---------------------------------------------------------------------------

def run_eval(model, device, val_csv_path, images_base, k_values=(1, 5, 10)):
    """Строим flat-индекс из val-каталога и оцениваем."""
    from PIL import Image
    from visual_search.evaluation.val_dataset import ValDataset
    from visual_search.evaluation.metrics import aggregate, ModeMetrics
    from visual_search.index.ann import ANNIndex, IndexSpec
    import pandas as pd

    dataset = ValDataset(images_base=images_base)

    # Собираем каталог
    image_id_to_path: dict[int, str] = {}
    for i in range(len(dataset)):
        q = dataset[i]
        if q.image_id is not None and q.image_path is not None:
            image_id_to_path[q.image_id] = q.image_path
        for tid in q.target_image_ids:
            pass  # пути будем брать из CSV ниже

    df = pd.read_csv(val_csv_path)
    for _, row in df.iterrows():
        iid = int(row["image_id"]) if pd.notna(row.get("image_id")) else None
        ipath = str(row["image_path"]) if pd.notna(row.get("image_path")) else None
        if iid and ipath:
            image_id_to_path[iid] = ipath

    catalog_ids = sorted(image_id_to_path.keys())

    # Кодируем каталог
    model.eval()
    catalog_vecs, valid_ids = [], []
    with torch.no_grad():
        for start in range(0, len(catalog_ids), 64):
            batch_ids = catalog_ids[start: start + 64]
            tensors, ok_ids = [], []
            for iid in batch_ids:
                img_path = os.path.join(images_base, image_id_to_path[iid])
                try:
                    img = Image.open(img_path).convert("RGB")
                    t = model.preprocess_image(img).squeeze(0)
                    tensors.append(t)
                    ok_ids.append(iid)
                except Exception:
                    pass
            if not tensors:
                continue
            emb = model.encode_image(torch.stack(tensors).to(device))
            catalog_vecs.append(emb.cpu().numpy())
            valid_ids.extend(ok_ids)

    vectors = np.concatenate(catalog_vecs).astype(np.float32)
    index = ANNIndex(embed_dim=model.embed_dim, spec=IndexSpec(backend="flat"))
    index.build(vectors, np.array(valid_ids, dtype=np.int64))

    def search_fn(query):
        vecs, weights = [], []
        if query.image_path is not None and query.mode in ("image", "multimodal"):
            try:
                img = Image.open(query.image_path).convert("RGB")
                t = model.preprocess_image(img).to(device)
                with torch.no_grad():
                    v = model.encode_image(t).squeeze(0).cpu().numpy()
                vecs.append(v)
                weights.append(0.5 if query.mode == "multimodal" else 1.0)
            except Exception:
                pass
        if query.txt_query is not None and query.mode in ("txt", "multimodal"):
            tokens = model.tokenize(query.txt_query).to(device)
            with torch.no_grad():
                v = model.encode_text(tokens).squeeze(0).cpu().numpy()
            vecs.append(v)
            weights.append(0.5 if query.mode == "multimodal" else 1.0)
        if not vecs:
            return []
        vec = np.average(vecs, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return [iid for iid, _ in index.search(vec, k=max(k_values))]

    # Оцениваем по режимам
    results: dict[str, ModeMetrics] = {}
    all_r, all_t, all_c = [], [], []
    for mode in ("image", "txt", "multimodal"):
        queries = dataset.get_by_mode(mode)
        if not queries:
            continue
        ranks, targets, cats = [], [], []
        for q in queries:
            ranked = search_fn(q)
            ranks.append(ranked)
            targets.append(q.target_image_ids)
            cats.append(str(q.metadata.get("param2") or "unknown"))
        results[mode] = aggregate(ranks, targets, list(k_values), cats, mode=mode)
        all_r += ranks; all_t += targets; all_c += cats
    results["all"] = aggregate(all_r, all_t, list(k_values), all_c, mode="all")
    return results


def print_comparison(results, baseline=BASELINE):
    print(f"\n{'Mode':<12} {'N':>5}  {'R@10':>6}  {'MRR':>6}  {'ΔMRR':>7}")
    print("-" * 48)
    for mode in ("image", "txt", "multimodal", "all"):
        if mode not in results:
            continue
        m = results[mode]
        mrr = m.mrr_score
        r10 = m.recall_at_k.get(10, 0)
        base_mrr = baseline.get(mode, {}).get("mrr", 0)
        delta = mrr - base_mrr
        sign = "+" if delta >= 0 else ""
        print(f"{mode:<12} {m.count:>5}  {r10:>6.3f}  {mrr:>6.3f}  {sign}{delta:>6.3f}")
    print("-" * 48)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    log_every: int = 20,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.perf_counter()
    images_processed = 0

    for step, (images, tokens, _texts) in enumerate(loader, 1):
        images = images.to(device)
        tokens = tokens.to(device)

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(tokens)
        loss = loss_fn(img_emb, txt_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        images_processed += images.size(0)

        if step % log_every == 0:
            elapsed = time.perf_counter() - t_start
            imgs_per_sec = images_processed / elapsed
            avg_loss = total_loss / n_batches
            log.info(
                "Epoch %d  step %d/%d  loss=%.4f  %.0f img/s",
                epoch, step, len(loader), avg_loss, imgs_per_sec,
            )

    elapsed = time.perf_counter() - t_start
    return {
        "avg_loss": total_loss / max(n_batches, 1),
        "steps": n_batches,
        "elapsed_sec": elapsed,
        "imgs_per_sec": images_processed / elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-parquet", default="data/interim/mini_train.parquet")
    parser.add_argument("--images-base", default="data/raw/dataset_1M")
    parser.add_argument("--val-csv", default=None,
                        help="Путь к val_dataset.csv (None = встроенный)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 = синхронно (безопаснее на MPS)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Лимит шагов per epoch (для быстрой проверки)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Пропустить eval после каждой эпохи (быстрее)")
    parser.add_argument("--out-dir", default="experiments/finetune_mini")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    # --- Device ---
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    # --- Модель ---
    from visual_search.models.registry import build_model
    from visual_search.models import encoders  # noqa — регистрация

    log.info("Загружаем xlm_clip_vit_b32 ...")
    model = build_model({"name": "xlm_clip_vit_b32"}).to(device)
    log.info("embed_dim=%d", model.embed_dim)

    # --- Процессор для датасета ---
    preprocess, tokenizer = model.get_processor()

    # --- Данные ---
    log.info("Загружаем %s ...", args.train_parquet)
    loader = build_train_loader(
        parquet_path=args.train_parquet,
        images_root=args.images_base,
        preprocess=preprocess,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log.info("Train loader: %d batches/epoch (batch=%d)", len(loader), args.batch_size)

    # --- Loss ---
    from visual_search.models.losses import InfoNCELoss
    loss_fn = InfoNCELoss().to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    # Cosine LR
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    # --- Val CSV ---
    val_csv = args.val_csv or str(
        Path(__file__).parent.parent
        / "src/visual_search/evaluation/val_dataset/val_dataset.csv"
    )

    # --- Выход ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log = {
        "model": "xlm_clip_vit_b32",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_items": len(loader.dataset),
        "baseline": BASELINE,
        "epochs_log": [],
    }

    log.info("=" * 55)
    log.info("Zero-shot baseline (для сравнения):")
    for mode, m in BASELINE.items():
        log.info("  [%s]  MRR=%.3f  R@10=%.3f", mode, m["mrr"], m["recall@10"])
    log.info("=" * 55)

    # --- Цикл обучения ---
    for epoch in range(1, args.epochs + 1):
        log.info("\n--- Epoch %d/%d ---", epoch, args.epochs)

        # Ограничение шагов для быстрого теста
        if args.max_steps:
            from itertools import islice

            class _LimitedLoader:
                """Обёртка над islice с поддержкой len() для логирования."""
                def __init__(self, loader, n):
                    self._loader = loader
                    self._n = n
                def __iter__(self):
                    return islice(iter(self._loader), self._n)
                def __len__(self):
                    return self._n

            train_stats = train_epoch(
                model, _LimitedLoader(loader, args.max_steps),
                optimizer, loss_fn, device, epoch,
            )
        else:
            train_stats = train_epoch(
                model, loader, optimizer, loss_fn, device, epoch,
            )
        scheduler.step()

        log.info(
            "Epoch %d done: loss=%.4f  %.0f img/s  %.1f min",
            epoch,
            train_stats["avg_loss"],
            train_stats["imgs_per_sec"],
            train_stats["elapsed_sec"] / 60,
        )

        epoch_entry = {"epoch": epoch, "train": train_stats}

        # --- Eval ---
        if not args.no_eval:
            log.info("Запускаем eval ...")
            model.eval()
            eval_results = run_eval(
                model, device, val_csv, args.images_base,
                k_values=tuple(args.k_values),
            )
            log.info("Результаты после epoch %d:", epoch)
            print_comparison(eval_results)

            epoch_entry["eval"] = {
                mode: m.as_flat_dict() for mode, m in eval_results.items()
            }

        run_log["epochs_log"].append(epoch_entry)

        # Сохраняем checkpoint
        ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_stats": train_stats,
        }, ckpt_path)
        log.info("Checkpoint: %s", ckpt_path)

    # --- Итоговый лог ---
    log_path = out_dir / "run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)
    log.info("\nЛог сохранён: %s", log_path)

    log.info("\n=== ИТОГ ===")
    log.info("Обучение завершено. Скорость: ~%.0f img/s на %s",
             run_log["epochs_log"][-1]["train"]["imgs_per_sec"], device)
    if run_log["epochs_log"] and "eval" in run_log["epochs_log"][-1]:
        final_eval = run_log["epochs_log"][-1]["eval"]
        log.info("Финальные метрики:")
        for mode in ("txt", "multimodal", "all"):
            if mode in final_eval:
                mrr = final_eval[mode].get("mrr", 0)
                base = BASELINE.get(mode, {}).get("mrr", 0)
                delta = mrr - base
                sign = "+" if delta >= 0 else ""
                log.info("  [%s] MRR=%.3f  (baseline=%.3f  %s%.3f)",
                         mode, mrr, base, sign, delta)


if __name__ == "__main__":
    main()
