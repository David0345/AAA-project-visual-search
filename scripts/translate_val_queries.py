#!/usr/bin/env python3
"""Перевести RU→EN текстовые запросы val_dataset (txt + multimodal) для англоязычных
моделей (Marqo-FashionSigLIP). Кэш: {query_id: english_text} → JSON.

Бэкенды:
  nllb  — facebook/nllb-200-distilled-600M (сильнее на лексике, по умолчанию)
  marian— Helsinki-NLP/opus-mt-ru-en (лёгкий, но грубый на fashion-словах)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visual_search.common.io import PROJECT_ROOT

VAL = PROJECT_ROOT / "src/visual_search/evaluation/val_dataset/val_dataset.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["nllb", "marian"], default="nllb")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/interim/val_txt_en_nllb.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.backend == "nllb":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name, src_lang="rus_Cyrl")
        mt = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
        bos = tok.convert_tokens_to_ids("eng_Latn")
    else:
        from transformers import MarianMTModel, MarianTokenizer
        name = "Helsinki-NLP/opus-mt-ru-en"
        tok = MarianTokenizer.from_pretrained(name)
        mt = MarianMTModel.from_pretrained(name).to(device).eval()
        bos = None

    val = pd.read_csv(VAL)
    rows = val[val["mode"].isin(["txt", "multimodal"])][["query_id", "txt_query"]].dropna()
    qids = rows["query_id"].astype(int).tolist()
    texts = rows["txt_query"].astype(str).tolist()
    print(f"[{args.backend}] переводим {len(texts)} запросов на {device} ...")

    out, B = {}, 32
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        kw = {"forced_bos_token_id": bos} if bos is not None else {}
        with torch.no_grad():
            gen = mt.generate(**enc, max_length=64, **kw)
        for qid, e in zip(qids[i:i + B], tok.batch_decode(gen, skip_special_tokens=True)):
            out[str(qid)] = e

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"сохранено: {args.out} ({len(out)})")
    for qid, ru in list(zip(qids, texts))[:6]:
        print(f"  {ru!r} -> {out[str(qid)]!r}")


if __name__ == "__main__":
    main()
