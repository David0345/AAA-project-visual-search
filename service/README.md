# Avito Visual Search — сервис

Мультимодальный поиск по каталогу одежды: режимы `image` (по фото), `txt` (по тексту),
`multimodal` (фото + текст-модификатор). Bi-encoder (SigLIP 2, дообучен на синтетике) +
FAISS-индекс каталога. Держит целевую нагрузку 1 RPS (согласовать с куратором).

## Структура
```
service/
  app/
    config.py    — настройки через env (VS_*), без магических констант
    schemas.py   — pydantic-схемы запроса/ответа
    engine.py    — загрузка модели+индекса, кодирование запроса, поиск
    main.py      — FastAPI: POST /search, GET /health
  Dockerfile
  requirements.txt
  tests/
artifacts/        — catalog.faiss + catalog_meta.parquet + checkpoint (готовятся офлайн)
```

## Артефакты (готовятся до сборки)
- `artifacts/catalog.faiss` — FAISS-индекс эмбеддингов каталога (строка i ↔ товар i).
- `artifacts/catalog_meta.parquet` — `item_id, image_path` в том же порядке.
- `artifacts/model.pt` — чекпойнт дообученной модели (`VS_CHECKPOINT_PATH`).
Сборка индекса: `python scripts/build_catalog_index.py` (см. репозиторий).

## Запуск (воспроизведение для жюри)
```bash
# 1) клон
git clone <repo> && cd AAA-project-visual-search

# 2) сборка контейнера
docker build -f service/Dockerfile -t avito-search .

# 3) запуск (артефакты монтируются в /app/artifacts)
docker run -p 8080:8080 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e VS_CHECKPOINT_PATH=/app/artifacts/model.pt \
  avito-search

# 4) пример запроса → 200
curl -s http://localhost:8080/health
curl -s -X POST http://localhost:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"mode":"txt","text":"красное вечернее платье","top_k":5}'
```
Ответ: `{"mode":"txt","hits":[{"item_id":..., "score":...}, ...]}`.

## Конфигурация (env, префикс `VS_`)
`VS_MODEL_NAME`, `VS_CHECKPOINT_PATH`, `VS_INDEX_PATH`, `VS_META_PATH`,
`VS_TOP_K`, `VS_MM_IMAGE_WEIGHT`, `VS_DEVICE` (cpu|cuda), `VS_PORT`.

## Тесты
```bash
pytest service/tests
```
