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
from visual_search.common.seed import set_seed
from visual_search.common.io import PROJECT_ROOT, RAW_DIR, INTERIM_DIR, EXPERIMENTS_DIR

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

class _CategoryBatchSampler:
    """Hard-negative батчинг: товары одной категории (param2) попадают в один
    батч → in-batch негативы становятся «трудными» (похожие товары), модель учится
    их различать. Каждую эпоху перетасовка внутри категорий и порядка батчей."""

    def __init__(self, categories, batch_size, seed=42, drop_last=True):
        self.groups: dict[str, list[int]] = {}
        for i, c in enumerate(categories):
            self.groups.setdefault(c, []).append(i)
        self.bs = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._n = len(categories) // batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order: list[int] = []
        for k in rng.permutation(list(self.groups.keys())):
            idxs = self.groups[k][:]
            rng.shuffle(idxs)
            order.extend(idxs)
        batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        if self.drop_last and batches and len(batches[-1]) < self.bs:
            batches.pop()
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self._n


def build_train_loader(
    parquet_path: str,
    images_root: str,
    preprocess,
    tokenizer,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    hard_neg: bool = False,
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

            # Мульти-фото режим: если есть image_paths (титул + доп.ракурсы),
            # каждый вызов сэмплируем случайный ракурс → визуальная аугментация
            # по эпохам без ложных негативов (в батче 1 строка = 1 товар).
            self.multi_image = "image_paths" in self.df.columns
            if self.multi_image and self.df["image_paths"].dtype == object:
                self.df["image_paths"] = self.df["image_paths"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            # категория товара (param2) для hard-negative батчинга
            self.categories = (
                self.df["param2"].fillna("unknown").astype(str).values
                if "param2" in self.df.columns else None
            )
            log.info("Dataset: %d items (multi_image=%s)", len(self.df), self.multi_image)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            # Изображение: в мульти-фото режиме случайный ракурс, иначе титул
            rel_path = row["title_image_path"]
            if self.multi_image:
                paths = row["image_paths"]
                if isinstance(paths, (list, tuple, np.ndarray)) and len(paths) > 0:
                    rel_path = paths[int(self.rng.integers(len(paths)))]

            img_path = self.images_root / rel_path
            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(image)
            except Exception as e:
                log.debug("Ошибка загрузки %s: %s", img_path, e)
                try:
                    image = Image.open(self.images_root / row["title_image_path"]).convert("RGB")
                    img_tensor = self.preprocess(image)
                except Exception:
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
    if hard_neg and ds.categories is not None:
        sampler = _CategoryBatchSampler(ds.categories, batch_size, seed)
        log.info("Hard-negative батчинг ВКЛ (категорийные батчи, %d батчей)", len(sampler))
        return DataLoader(ds, batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=False, collate_fn=collate)
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
    amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    scheduler=None,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.perf_counter()
    images_processed = 0

    for step, (images, tokens, _texts) in enumerate(loader, 1):
        images = images.to(device)
        tokens = tokens.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(tokens)
            loss = loss_fn(img_emb, txt_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # per-step (warmup + cosine)

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

def append_ledger(run_log: dict, ledger_path: Path) -> None:
    """Дописать одну строку с итогом прогона в общий JSONL-ledger.

    Цель — копить историю всех замеров (config + финальные метрики по режимам),
    чтобы отслеживать, не деградирует ли txt после дообучения (как было у коллег).
    """
    import datetime as _dt

    epochs_log = run_log.get("epochs_log", [])
    last = epochs_log[-1] if epochs_log else {}
    final_eval = last.get("eval")  # None если запускали с --no-eval

    def _mode_metric(mode: str, key: str):
        if not final_eval or mode not in final_eval:
            return None
        return final_eval[mode].get(key)

    # Пер-эпоховые траектории txt/all MRR + лучшая эпоха по all MRR
    def _traj(mode):
        return [(e.get("eval") or {}).get(mode, {}).get("mrr") for e in epochs_log]
    all_traj = _traj("all")
    best_epoch = None
    valid = [(i + 1, v) for i, v in enumerate(all_traj) if v is not None]
    if valid:
        best_epoch = max(valid, key=lambda t: t[1])[0]

    freeze = [k.split("_")[1] for k in ("freeze_text", "freeze_visual", "freeze_backbone")
              if run_log.get(k)]
    rec = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "run_name": run_log.get("run_name"),
        "model": run_log.get("model"),
        "device": run_log.get("device"),
        "amp": run_log.get("amp"),
        "epochs": run_log.get("epochs"),
        "batch_size": run_log.get("batch_size"),
        "lr": run_log.get("lr"),
        "loss": run_log.get("loss"),
        "temperature": run_log.get("temperature"),
        "freeze": "+".join(freeze) if freeze else None,
        "grad_checkpointing": run_log.get("grad_checkpointing"),
        "warmup_frac": run_log.get("warmup_frac"),
        "train_items": run_log.get("train_items"),
        "imgs_per_sec": round(last.get("train", {}).get("imgs_per_sec", 0), 1) if last else None,
        "loss_trajectory": [round(e.get("train", {}).get("avg_loss", float("nan")), 4)
                            for e in epochs_log],
        "txt_mrr_trajectory": [round(v, 4) if v is not None else None for v in _traj("txt")],
        "mm_mrr_trajectory": [round(v, 4) if v is not None else None for v in _traj("multimodal")],
        "all_mrr_trajectory": [round(v, 4) if v is not None else None for v in all_traj],
        "best_epoch": best_epoch,
        "eval": None,
    }
    if final_eval:
        rec["eval"] = {
            mode: {"mrr": _mode_metric(mode, "mrr"),
                   "recall@10": _mode_metric(mode, "recall@10")}
            for mode in ("image", "txt", "multimodal", "all")
            if mode in final_eval
        }
        # Явный флаг деградации txt относительно zero-shot baseline
        base_txt = run_log.get("baseline", {}).get("txt", {}).get("mrr")
        ft_txt = _mode_metric("txt", "mrr")
        if base_txt is not None and ft_txt is not None:
            rec["txt_mrr_delta_vs_baseline"] = round(ft_txt - base_txt, 4)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Ledger += %s  (%s)", ledger_path, rec.get("run_name"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="xlm_clip_vit_b32",
                        help="имя модели из registry (xlm_clip_vit_b32, siglip2_b16_256, siglip2_l16_256, ...)")
    parser.add_argument("--train-parquet", default=INTERIM_DIR / "mini_train.parquet")
    parser.add_argument("--images-base", default=RAW_DIR / "dataset_1M")
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Mixed precision (bf16 autocast). По умолчанию вкл. на cuda")
    parser.add_argument("--run-name", default=None,
                        help="Метка прогона для ledger (по умолчанию = имя out-dir)")
    # --- Рычаги оптимизации ---
    parser.add_argument("--loss", default="infonce", choices=["infonce", "sigmoid"],
                        help="Контрастивный лосс")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Начальная температура лосса")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Заморозить текстовую башню (защищает txt)")
    parser.add_argument("--freeze-visual", action="store_true",
                        help="Заморозить визуальную башню")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Заморозить обе башни (учится только logit_scale/проекции)")
    parser.add_argument("--hard-neg-batching", action="store_true",
                        help="батчи из товаров одной категории (param2) → трудные in-batch негативы")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="Gradient checkpointing (экономит память для больших batch)")
    parser.add_argument("--warmup-frac", type=float, default=0.0,
                        help="Доля шагов на linear warmup перед cosine (0 = без warmup)")
    parser.add_argument("--save-ckpt", default="all", choices=["none", "best", "all"],
                        help="none — не писать чекпойнты (свип); best — только лучшую эпоху (model-only); all — каждую")
    parser.add_argument("--out-dir", default=EXPERIMENTS_DIR / "finetune_mini")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    set_seed(42, deterministic=False)

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

    # AMP по умолчанию включён на cuda (bf16 — A800 поддерживает нативно, GradScaler не нужен)
    amp = (device.type == "cuda") if args.amp is None else args.amp
    if amp and device.type != "cuda":
        log.warning("AMP запрошен на %s — отключаю (поддержано только на cuda)", device.type)
        amp = False
    log.info("AMP (bf16 autocast): %s", amp)

    # --- Модель ---
    from visual_search.models.registry import build_model
    from visual_search.models import encoders  # noqa — регистрация

    model_config = {
        "name": args.model,
        "freeze_text": args.freeze_text,
        "freeze_visual": args.freeze_visual,
        "freeze_backbone": args.freeze_backbone,
        "grad_checkpointing": args.grad_checkpointing,
    }
    log.info("Загружаем %s (config=%s) ...", args.model,
             {k: v for k, v in model_config.items() if v and k != "name"})
    model = build_model(model_config).to(device)
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
        hard_neg=args.hard_neg_batching,
    )
    log.info("Train loader: %d batches/epoch (batch=%d)", len(loader), args.batch_size)

    # --- Loss ---
    from visual_search.models.losses import InfoNCELoss, SigmoidLoss
    if args.loss == "sigmoid":
        loss_fn = SigmoidLoss(temperature=args.temperature).to(device)
    else:
        loss_fn = InfoNCELoss(temperature=args.temperature).to(device)
    log.info("Loss: %s (temperature=%.3f)", args.loss, args.temperature)

    # --- Optimizer (только обучаемые параметры, чтобы замороженные не висели в AdamW) ---
    trainable = [p for p in model.parameters() if p.requires_grad] + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )

    # --- LR schedule: linear warmup -> cosine, шаг по батчу ---
    import math as _math
    steps_per_epoch = args.max_steps or len(loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_frac * total_steps)
    eta_ratio = 0.1  # eta_min = lr * 0.1

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        return eta_ratio + (1.0 - eta_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    log.info("Schedule: warmup=%d / total=%d шагов (warmup_frac=%.2f)",
             warmup_steps, total_steps, args.warmup_frac)

    # --- Val CSV ---
    val_csv = args.val_csv or str(PROJECT_ROOT / "src/visual_search/evaluation/val_dataset/val_dataset.csv")

    # --- Выход ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or out_dir.name
    run_log = {
        "run_name": run_name,
        "model": "xlm_clip_vit_b32",
        "device": str(device),
        "amp": amp,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "temperature": args.temperature,
        "freeze_text": args.freeze_text,
        "freeze_visual": args.freeze_visual,
        "freeze_backbone": args.freeze_backbone,
        "grad_checkpointing": args.grad_checkpointing,
        "warmup_frac": args.warmup_frac,
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
    best_metric = float("-inf")  # для --save-ckpt best (по all MRR, иначе -loss)
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
                optimizer, loss_fn, device, epoch, amp=amp, scheduler=scheduler,
            )
        else:
            train_stats = train_epoch(
                model, loader, optimizer, loss_fn, device, epoch, amp=amp, scheduler=scheduler,
            )

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

        # --- Checkpoint policy ---
        if args.save_ckpt == "all":
            ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_stats": train_stats,
            }, ckpt_path)
            log.info("Checkpoint: %s", ckpt_path)
        elif args.save_ckpt == "best":
            # Метрика выбора: all MRR (если есть eval), иначе -loss
            cur = None
            if "eval" in epoch_entry and "all" in epoch_entry["eval"]:
                cur = epoch_entry["eval"]["all"].get("mrr")
            if cur is None:
                cur = -train_stats["avg_loss"]
            if cur > best_metric:
                best_metric = cur
                for old in out_dir.glob("best_ep*_model_only.pt"):
                    old.unlink()
                best_path = out_dir / f"best_ep{epoch}_model_only.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),  # model-only (~1.4G), без optimizer
                    "train_stats": train_stats,
                    "eval_metrics": epoch_entry.get("eval"),
                    "run_name": run_name,
                }, best_path)
                log.info("Best checkpoint (model-only): %s  (metric=%.4f)", best_path, cur)

    # --- Итоговый лог ---
    log_path = out_dir / "run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)
    log.info("\nЛог сохранён: %s", log_path)

    # --- Append в общий ledger (append-only, чтобы накапливать историю замеров) ---
    append_ledger(run_log, EXPERIMENTS_DIR / "metrics_ledger.jsonl")

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
