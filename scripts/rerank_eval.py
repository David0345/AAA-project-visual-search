#!/usr/bin/env python3
"""Двухстадийный поиск: bi-encoder (SigLIP2) достаёт топ-K, Qwen2-VL переоценивает
релевантность кандидата запросу (pointwise 0-10) и пересортировывает. Qwen2-VL
понимает русский → реранкер кросс-модальный без перевода.

Меряем txt и multimodal на каталоге 52k: MRR/R@10 ДО (bi-encoder) и ПОСЛЕ rerank.
image-режим не реранкаем (он и так лучший; image-image другой механизм).

Пример:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/rerank_eval.py \
    --model siglip2_l16_256 --ckpt experiments/l16_synth_lr1e-6/best_ep2_model_only.pt \
    --catalog-size 50000 --top-k 30 --modes txt multimodal
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))                       # eval_full
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader

from eval_full import build_catalog, load_model, _CatalogDS
from visual_search.common.io import RAW_DIR
from visual_search.evaluation.val_dataset import ValDataset
from visual_search.evaluation.metrics import aggregate
from visual_search.index.ann import ANNIndex, IndexSpec

QWEN = "Qwen/Qwen2-VL-7B-Instruct"
SCORE_PROMPT = (
    "Запрос покупателя: «{q}». Оцени, насколько товар на фото подходит под этот запрос. "
    "Ответь ТОЛЬКО одним числом от 0 до 10 (10 — идеально подходит, 0 — совсем не подходит)."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2_l16_256")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--catalog-size", type=int, default=50000)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--modes", nargs="+", default=["txt", "multimodal"])
    ap.add_argument("--mm-image-weight", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--score-batch", type=int, default=16)
    ap.add_argument("--images-base", default=str(RAW_DIR / "dataset_1M"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device(args.device)
    K = max(args.top_k, 10)

    # --- 1. bi-encoder: каталог + индекс ---
    model = load_model(args.model, args.ckpt, device)
    preprocess, _ = model.get_processor()
    catalog = build_catalog(args.images_base, args.catalog_size, args.seed)
    ids = list(catalog.keys()); paths = [catalog[i] for i in ids]
    print(f"кодируем каталог ({len(ids)}) bi-encoder'ом ...")
    dl = DataLoader(_CatalogDS(ids, paths, args.images_base, preprocess),
                    batch_size=256, num_workers=args.num_workers, shuffle=False)
    vecs, vids = [], []
    with torch.no_grad():
        for imgs, bids in dl:
            keep = bids != -1
            if keep.sum() == 0:
                continue
            vecs.append(model.encode_image(imgs[keep].to(device)).cpu().numpy())
            vids.extend(bids[keep].tolist())
    vectors = np.concatenate(vecs).astype(np.float32)
    index = ANNIndex(embed_dim=model.embed_dim, spec=IndexSpec(backend="flat"))
    index.build(vectors, np.array(vids, dtype=np.int64))
    id2path = dict(catalog)

    def retrieve(q):
        vl, w = [], []
        if q.image_path is not None and q.mode in ("image", "multimodal"):
            try:
                t = model.preprocess_image(Image.open(q.image_path).convert("RGB")).to(device)
                with torch.no_grad():
                    vl.append(model.encode_image(t).squeeze(0).cpu().numpy())
                w.append(args.mm_image_weight if q.mode == "multimodal" else 1.0)
            except Exception:
                pass
        if q.txt_query is not None and q.mode in ("txt", "multimodal"):
            with torch.no_grad():
                vl.append(model.encode_text(model.tokenize(q.txt_query).to(device)).squeeze(0).cpu().numpy())
            w.append((1 - args.mm_image_weight) if q.mode == "multimodal" else 1.0)
        if not vl:
            return []
        v = np.average(vl, axis=0, weights=w).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        return [iid for iid, _ in index.search(v, k=200)]

    # --- 2. Qwen2-VL reranker ---
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    qproc = AutoProcessor.from_pretrained(QWEN, max_pixels=200704)

    def qwen_scores(query_text, cand_paths):
        scores = []
        for s in range(0, len(cand_paths), args.score_batch):
            chunk = cand_paths[s:s + args.score_batch]
            msgs = []
            for p in chunk:
                img = Image.open(Path(args.images_base) / p).convert("RGB")
                msgs.append([{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": SCORE_PROMPT.format(q=query_text)}]}])
            texts = [qproc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            imgs_in, _ = process_vision_info(msgs)
            inp = qproc(text=texts, images=imgs_in, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = qwen.generate(**inp, max_new_tokens=5, do_sample=False)
            for i, o in zip(inp.input_ids, out):
                dec = qproc.tokenizer.decode(o[len(i):], skip_special_tokens=True)
                num = "".join(c for c in dec if c.isdigit())
                scores.append(float(num[:2]) if num else 0.0)
        return scores

    # --- 3. eval ДО и ПОСЛЕ rerank ---
    ds = ValDataset(images_base=args.images_base)
    results = {}
    for mode in args.modes:
        qs = ds.get_by_mode(mode)
        base_ranks, rr_ranks, targets, cats = [], [], [], []
        t0 = time.time()
        for j, q in enumerate(qs):
            ranked = retrieve(q)
            base_ranks.append(ranked)
            topk = ranked[:K]
            qtext = q.txt_query or ""
            cand_paths = [id2path[i] for i in topk if i in id2path]
            valid = [i for i in topk if i in id2path]
            sc = qwen_scores(qtext, cand_paths)
            order = sorted(range(len(valid)), key=lambda x: -sc[x])
            rr = [valid[o] for o in order] + ranked[K:]
            rr_ranks.append(rr)
            targets.append(q.target_image_ids)
            cats.append(str(q.metadata.get("param2") or "unknown"))
            if j % 20 == 0:
                print(f"  [{mode}] {j}/{len(qs)}  {(j+1)/(time.time()-t0+1e-9):.2f} q/s", flush=True)
        base = aggregate(base_ranks, targets, [1, 5, 10], cats, mode=mode)
        rr = aggregate(rr_ranks, targets, [1, 5, 10], cats, mode=mode)
        results[mode] = (base, rr)
        print(f"\n=== {mode} (top-{K} rerank) ===")
        print(f"  ДО   : MRR={base.mrr_score:.3f}  R@10={base.recall_at_k.get(10,0):.3f}")
        print(f"  ПОСЛЕ: MRR={rr.mrr_score:.3f}  R@10={rr.recall_at_k.get(10,0):.3f}")


if __name__ == "__main__":
    main()
