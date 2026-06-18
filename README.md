# AAA-project-visual-search

## Основная информация

Цель проекта — микросервис визуального поиска по каталогу товаров Avito (~1 млн, одежда).
Три режима запроса: **текст**, **картинка**, **текст + картинка**. Модель — SigLIP 2
(ViT-L/16), дообученная контрастивно на синтетических запросах; поиск по каталогу — FAISS.
Сервис рассчитан на нагрузку ≥ 1 rps.

### Команда «CLIPовое мышление»
- Барсегян Давид (Капитан)
- Васютин Павел
- Чернова Анна

## Структура репозитория
```
src/visual_search/
  serving/        сервис FastAPI: app.py, search.py (движок), schemas.py, index.html (фронт)
  models/         энкодеры (open_clip SigLIP2/CLIP), лоссы, registry
  index/          ANNIndex — обёртка FAISS: build / save / load / search
  data, training, evaluation   пайплайн данных, обучения, оценки
scripts/          CLI: serve.py (запуск сервиса), сборка индекса, обучение, оценка, нагрузочный тест
experiments/      лог исследования (SUMMARY.md — итог и метрики)
Dockerfile, requirements-serving.txt   контейнеризация сервиса
```

## Запуск сервиса в Docker

Артефакты монтируются томом в `/artifacts`:
- `model.pt` — веса дообученной модели;
- `serving_index/` — FAISS-индекс (`index.faiss`, `ids.npy`, `meta.json`, `metadata.parquet`);
- `catalog_images/` — картинки каталога (`images/.../*.jpg`) для отображения.

```bash
# 1) клон
git clone https://github.com/David0345/AAA-project-visual-search.git
cd AAA-project-visual-search

# 2) сборка образа
docker build -t avito-search .

# 3) запуск (CPU). На GPU-хосте добавить: --gpus all -e DEVICE=cuda
docker run -d -p 8000:8000 \
  -v /path/to/artifacts:/artifacts \
  -e PUBLIC_BASE_URL=http://<HOST>:8000/files \
  avito-search

# 4) пример запроса -> ответ с кодом 200
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/search \
  -F "text=красное вечернее платье" -F "top_k=5"
```
Веб-интерфейс: `http://<HOST>:8000/`.

### API
- `GET  /health` — статус и размер каталога.
- `POST /api/search` (multipart): `text?`, `image?` (файл), `top_k`(=10)
  → `{status, query_mode, results:[{item_id, score, image_url, title, param2, brand}]}`.
- `GET  /` — фронтенд.

### Переменные окружения
`MODEL_DIR` (имя модели в registry), `CHECKPOINT` (веса), `INDEX_DIR`, `IMAGES_DIR`,
`DEVICE` (cpu|cuda), `IMAGE_URL_MODE` (public|presign), `PUBLIC_BASE_URL`, `HOST`, `PORT`.

## Сборка индекса каталога
`scripts/build_catalog_index.py` — кодирование каталога моделью в FAISS-индекс;
`scripts/repack_index_for_serving.py` — упаковка в формат сервиса.

## Нагрузочное тестирование
`scripts/loadtest.py --url http://<HOST>:8000/api/search --rps 1 --duration 60` —
разнообразные запросы без кэширования, отчёт по throughput и p50/p95/p98.

## ML-часть
Выбор и дообучение модели, эксперименты — в `scripts/` и `experiments/SUMMARY.md`.
