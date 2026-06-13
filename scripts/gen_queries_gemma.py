#!/usr/bin/env python3
"""Генерация естественных поисковых запросов из метаданных товара через gemma-2-9b-it.
Заменяет шаблонные queries (синонимы/транслит) на более «человеческие» и разнообразные,
чтобы пробить потолок дообучения (шаблонные запросы слишком узкие).

Режимы:
  --test N      сгенерировать для N товаров и вывести (проверка качества)
  --full --out  сгенерировать для всех/части и сохранить parquet (item_id, queries_llm)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import INTERIM_DIR

MODEL = "google/gemma-2-9b-it"

SYS = (
    "Ты — покупатель на Авито, ищешь женскую одежду. По описанию товара придумай "
    "РОВНО {n} разных коротких поисковых запросов, как реально пишут люди в строке поиска. "
    "Разная длина и формулировки: где-то 2 слова, где-то с деталями. Только русский. "
    "Не повторяй одно и то же. Без нумерации и пояснений — каждый запрос с новой строки."
)


def build_prompt(row) -> str:
    parts = []
    for k, label in [("predmet_odezhdy", "предмет"), ("param2", "категория"),
                     ("cvet", "цвет"), ("brand", "бренд"), ("sostoyanie", "состояние")]:
        v = row.get(k)
        if pd.notna(v) and str(v).strip() and str(v).lower() not in ("без бренда", "другой", "nan"):
            parts.append(f"{label}: {v}")
    return "Товар — " + ", ".join(parts) if parts else "Товар — женская одежда"


def parse_lines(text: str, n: int) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip(" -•0123456789.)\t").strip()
        if ln and len(ln) > 1:
            out.append(ln)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число товаров (full)")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-parquet", default=str(INTERIM_DIR / "train_full.parquet"))
    ap.add_argument("--out", default=str(INTERIM_DIR / "queries_llm.parquet"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    df = pd.read_parquet(args.train_parquet)
    if args.test:
        df = df.sample(args.test, random_state=args.seed)
    elif args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    print(f"генерим для {len(rows)} товаров, n_queries={args.n_queries}")

    def gen_batch(prompts):
        msgs = [[{"role": "user", "content": SYS.format(n=args.n_queries) + "\n\n" + p}] for p in prompts]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.9, top_p=0.95)
        gen = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    results, t0 = [], time.time()
    B = args.batch_size
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        prompts = [build_prompt(r) for r in batch]
        outs = gen_batch(prompts)
        for r, p, o in zip(batch, prompts, outs):
            qs = parse_lines(o, args.n_queries)
            results.append({"item_id": int(r["item_id"]), "queries_llm": qs})
            if args.test:
                print(f"\n--- {p}")
                for q in qs:
                    print(f"     • {q}")
        if not args.test and i % (B * 10) == 0:
            done = i + len(batch)
            rate = done / (time.time() - t0 + 1e-9)
            print(f"  {done}/{len(rows)}  {rate:.1f} item/s  ETA {(len(rows)-done)/rate/60:.0f}min")

    if args.full:
        pd.DataFrame(results).to_parquet(args.out, index=False)
        print(f"сохранено: {args.out} ({len(results)})")


if __name__ == "__main__":
    main()
