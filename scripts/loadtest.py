#!/usr/bin/env python3
"""Нагрузочный тест сервиса поиска (для критерия «держит нагрузку»).
Шлёт разнообразные запросы БЕЗ кэширования на целевом RPS, считает throughput и
перцентили латентности (p50/p95/p98). Клиент — не зависит от реализации бэкенда,
бьёт по HTTP-эндпоинту.

Пример:
  python scripts/loadtest.py --url http://HOST:8080/search --rps 1 --duration 60 --mode txt
  python scripts/loadtest.py --url http://HOST:8080/search --rps 5 --duration 60 --mode image \
      --images-dir data/raw/dataset_1M/images
"""
from __future__ import annotations

import argparse, base64, glob, json, random, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

# разнообразные тексты (разные стили — против кэша и для реалистичности)
TEXTS = [
    "красное вечернее платье", "куртка зимняя женская", "джинсы мом высокая посадка",
    "сумка кожаная чёрная", "оверсайз толстовка", "льняная рубашка бежевая",
    "платье в цветочный принт", "пальто демисезон", "юбка плиссе миди",
    "кроссовки белые", "свитер вязаный v-образный вырез", "шорты джинсовые",
]


def sample_image_b64(images: list[str]) -> str:
    with open(random.choice(images), "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_request(mode: str, images: list[str], i: int) -> tuple[dict, dict]:
    """(data, files) для multipart-формы API. Текст уникализируем → без кэша."""
    data: dict = {"top_k": 20}
    files: dict = {}
    if mode in ("txt", "multimodal"):
        data["text"] = f"{random.choice(TEXTS)} {i}"
    if mode in ("image", "multimodal"):
        with open(random.choice(images), "rb") as f:
            files["image"] = ("q.jpg", f.read(), "image/jpeg")
    return data, files


def one_request(url: str, data: dict, files: dict, timeout: float) -> tuple[int, float]:
    t0 = time.perf_counter()
    try:
        r = requests.post(url, data=data, files=(files or None), timeout=timeout)
        return r.status_code, time.perf_counter() - t0
    except Exception:
        return 0, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--duration", type=int, default=60, help="секунд")
    ap.add_argument("--mode", choices=["txt", "image", "multimodal"], default="txt")
    ap.add_argument("--images-dir", default="data/raw/dataset_1M/images")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    images: list[str] = []
    if args.mode in ("image", "multimodal"):
        images = glob.glob(f"{args.images_dir}/**/*.jpg", recursive=True)
        if not images:
            raise SystemExit(f"нет картинок в {args.images_dir}")

    n = int(args.rps * args.duration)
    interval = 1.0 / args.rps
    print(f"шлём {n} запросов на {args.url} | mode={args.mode} | target {args.rps} rps / {args.duration}s")

    latencies, codes = [], []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(8, int(args.rps * 2))) as ex:
        futures = []
        for i in range(n):
            target = start + i * interval
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            data, files = build_request(args.mode, images, i)
            futures.append(ex.submit(one_request, args.url, data, files, args.timeout))
        for f in futures:
            code, lat = f.result()
            codes.append(code); latencies.append(lat)

    wall = time.perf_counter() - start
    lat = np.array(latencies) * 1000  # мс
    ok = sum(c == 200 for c in codes)
    print("\n=== РЕЗУЛЬТАТ ===")
    print(f"запросов: {len(codes)} | 200: {ok} | ошибок: {len(codes)-ok}")
    print(f"throughput: {len(codes)/wall:.2f} rps ({len(codes)/wall*60:.0f} rpm)")
    print(f"латентность мс: p50={np.percentile(lat,50):.0f}  p95={np.percentile(lat,95):.0f}  "
          f"p98={np.percentile(lat,98):.0f}  max={lat.max():.0f}")


if __name__ == "__main__":
    main()
