FROM python:3.11-slim

WORKDIR /app

# системные зависимости для pillow/faiss
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# пакет моделей (src/visual_search) + код сервиса
COPY src /app/src
COPY service/app /app/app

# артефакты (модель-чекпойнт + faiss-индекс + метаданные) монтируются или копируются
# в /app/artifacts; пути задаются через переменные окружения VS_*
ENV PYTHONPATH=/app/src VS_INDEX_PATH=/app/artifacts/catalog.faiss \
    VS_META_PATH=/app/artifacts/catalog_meta.parquet VS_DEVICE=cpu

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
