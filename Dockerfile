# Сервис визуального поиска (FastAPI + open_clip SigLIP2 + FAISS).
# CUDA-torch: на GPU-хосте (docker run --gpus all, DEVICE=cuda) считает на GPU,
# на любом другом сервере — fallback на CPU. Артефакты (модель/индекс/картинки)
# монтируются томом в /artifacts (см. README).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY scripts/serve.py ./scripts/serve.py

ENV PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_DIR=siglip2_l16_256 \
    CHECKPOINT=/artifacts/model.pt \
    INDEX_DIR=/artifacts/serving_index \
    IMAGES_DIR=/artifacts/catalog_images \
    DEVICE=cpu \
    IMAGE_URL_MODE=public

EXPOSE 8000
CMD ["python", "scripts/serve.py"]
