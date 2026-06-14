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
import argparse, re, sys, time
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import INTERIM_DIR, RAW_DIR

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
IMAGES_BASE = RAW_DIR / "dataset_1M"

PROMPT = (
    "На фото женская одежда. Сгенерируй РОВНО {n} поисковых запросов на русском. "
    "Каждый запрос — в одном из 5 стилей ниже (по одному запросу на стиль). "
    "ВАЖНО: ровно {n} строк, по одной на стиль. Пиши ТОЛЬКО сам запрос. "
    "НЕ начинай строку со слова «Короткий/Стандартный/Детальный/Разговорный/Атрибутивный» "
    "или любого названия стиля, НЕ нумеруй, без двоеточий-заголовков и пояснений.\n"
    "Стили (примеры стиля даны для ориентира, не копируй их дословно):\n"
    "1. Короткий: 2-3 слова, ключевые признаки (как «вечернее платье», «оверсайз толстовка», «светлые бриджи»).\n"
    "2. Стандартный: цвет + тип + базовый фасон (как «платье синее свободное», «льняная рубашка бежевая»).\n"
    "3. Детальный: фактура, длина, вырез, рукава, принт, крой (как «серая кофта v-образный вырез», «рваные джинсы с потертостями»).\n"
    "4. Разговорный: естественная фраза покупателя (как «рубашка в клетку с коротким рукавом», «брюки шоколадного цвета»).\n"
    "5. Атрибутивный: перечисление через запятую (как «платье, синее, свободное{brand_attr}»).\n"
    "Описывай ТОЛЬКО то, что реально ВИДНО на фото, не выдумывай. {brand_hint} "
    "Запрещены слова «Авито», «фото», «бренд» (кроме стиля 5)."
)


def brand_fields(brand):
    """(brand_hint, brand_attr) — подсказка про бренд и хвост для атрибутивного стиля."""
    if pd.notna(brand) and str(brand).strip() and str(brand).lower() not in ("без бренда", "другой", "nan"):
        return f"Бренд «{brand}» — добавь его в стиль 5 и, по желанию, в стиль 1-2.", f", {brand}"
    return "", ""


_STYLE_LABELS = {"короткий", "стандартный", "детальный", "разговорный", "атрибутивный", "стиль", "стили", "запрос", "запросы"}


def parse_lines(text: str, n: int) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip(" -•*\t").strip()
        ln = re.sub(r"^\d+[.)]\s*", "", ln)                     # убрать «1.» / «2)»
        m = re.match(r"^([А-Яа-яЁё ]{3,15}?)\s*:\s*(.+)$", ln)  # убрать «Атрибутивный: ...»
        if m and m.group(1).strip().lower() in _STYLE_LABELS:
            ln = m.group(2).strip()
        ln = ln.rstrip(" .;,").strip()                          # юзеры не пишут точку в конце
        if not ln or len(ln) < 2:
            continue
        if ln.rstrip(":").strip().lower() in _STYLE_LABELS:     # пропустить голый лейбл
            continue
        if ln.lower() not in {o.lower() for o in out}:          # дедуп внутри товара
            out.append(ln)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--total", type=int, default=None, help="детерм. подвыборка N товаров (для шардинга)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
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
    elif args.total:                                   # детерм. подвыборка + шард
        df = df.sample(min(args.total, len(df)), random_state=args.seed).reset_index(drop=True)
        if args.num_shards > 1:
            df = df.iloc[args.shard_id::args.num_shards]
    elif args.limit:
        df = df.sample(args.limit, random_state=args.seed)
    rows = df.to_dict("records")
    print(f"генерим для {len(rows)} товаров (shard {args.shard_id}/{args.num_shards}), n_queries={args.n_queries}")

    def gen_batch(batch):
        msgs = []
        for r in batch:
            img = Image.open(IMAGES_BASE / r["title_image_path"]).convert("RGB")
            bh, ba = brand_fields(r.get("brand"))
            txt = PROMPT.format(n=args.n_queries, brand_hint=bh, brand_attr=ba)
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
        Path(args.out + ".done").write_text(str(len(results)))   # маркер для оркестратора
        print(f"сохранено: {args.out} ({len(results)})")


if __name__ == "__main__":
    main()
