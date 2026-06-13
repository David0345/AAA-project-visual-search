#!/usr/bin/env python3
"""Генерация естественных поисковых запросов ПО КАРТИНКЕ товара через Qwen2-VL-7B.
VLM видит реальные атрибуты (принт, фасон, длина, рукав, фактура), которых нет в
метаданных → новый сигнал, выровненный с изображением. Бренд подаём текстом
(на фото не виден). Цель — пробить потолок дообучения шаблонных запросов.

Режимы:
  --test N           сгенерировать для N товаров и вывести (проверка качества)
  --full --limit K   сгенерировать для K товаров и сохранить parquet
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import INTERIM_DIR, RAW_DIR

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
IMAGES_BASE = RAW_DIR / "dataset_1M"

PROMPT = (
    "На фото женская одежда. Придумай РОВНО {n} коротких поисковых запросов "
    "на русском, как реально пишут люди в поиске. Описывай ТОЛЬКО то, что реально ВИДНО, "
    "не выдумывай деталей: тип вещи, цвет, принт/узор, фасон (оверсайз, приталенное), "
    "длину, рукава, вырез, фактуру (вязаное, кожаное, джинсовое). {brand_hint} "
    "Разной длины: где-то 2 слова, где-то с деталями. НЕ пиши слова «Авито», «фото», «бренд». "
    "Только запросы, каждый с новой строки, без нумерации и пояснений."
)


def brand_hint(brand) -> str:
    if pd.notna(brand) and str(brand).strip() and str(brand).lower() not in ("без бренда", "другой", "nan"):
        return f"Бренд «{brand}» — добавь его в 1-2 запроса."
    return ""


def parse_lines(text: str, n: int) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip(" -•*0123456789.)\t").strip()
        if ln and len(ln) > 1 and ":" not in ln[:12]:
            out.append(ln)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=401408, help="ограничение токенов на картинку (~512x784)")
    ap.add_argument("--train-parquet", default=str(INTERIM_DIR / "train_full.parquet"))
    ap.add_argument("--out", default=str(INTERIM_DIR / "queries_qwen.parquet"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=args.max_pixels)

    df = pd.read_parquet(args.train_parquet)
    if args.test:
        df = df.sample(args.test, random_state=args.seed)
    elif args.limit:
        df = df.sample(args.limit, random_state=args.seed)
    rows = df.to_dict("records")
    print(f"генерим для {len(rows)} товаров, n_queries={args.n_queries}")

    def gen_batch(batch):
        msgs = []
        for r in batch:
            img = Image.open(IMAGES_BASE / r["title_image_path"]).convert("RGB")
            txt = PROMPT.format(n=args.n_queries, brand_hint=brand_hint(r.get("brand")))
            msgs.append([{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": txt}]}])
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        image_inputs, _ = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=True, temperature=0.8, top_p=0.9)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return processor.batch_decode(trimmed, skip_special_tokens=True)

    results, t0 = [], time.time()
    B = args.batch_size
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        try:
            outs = gen_batch(batch)
        except Exception as e:
            print(f"  батч {i} упал: {e}"); continue
        for r, o in zip(batch, outs):
            qs = parse_lines(o, args.n_queries)
            results.append({"item_id": int(r["item_id"]), "queries_llm": qs})
            if args.test:
                print(f"\n--- item {r['item_id']} | бренд={r.get('brand')} | {r.get('title_image_path')}")
                for q in qs:
                    print(f"     • {q}")
        if not args.test and (i // B) % 5 == 0:
            done = i + len(batch); rate = done / (time.time() - t0 + 1e-9)
            print(f"  {done}/{len(rows)}  {rate:.2f} item/s  ETA {(len(rows)-done)/rate/60:.0f}min", flush=True)

    if args.full:
        pd.DataFrame(results).to_parquet(args.out, index=False)
        print(f"сохранено: {args.out} ({len(results)})")


if __name__ == "__main__":
    main()
