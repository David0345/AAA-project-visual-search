# Сервис визуального поиска (FastAPI + open_clip SigLIP2 + FAISS).
# Самодостаточный образ: веса модели и индекс (~4 ГБ) НЕ в git — они скачиваются
# с HuggingFace Hub при сборке (ARG ARTIFACTS_REPO). Базовая SigLIP кэшируется в
# образ, поэтому рантайму сеть не нужна. Картинки галереи отдаёт демо-сервер
# (PUBLIC_BASE_URL). CUDA-torch: на GPU-хосте (--gpus all, DEVICE=cuda) — на GPU,
# иначе fallback на CPU. Чтобы подложить свои артефакты — смонтировать том в
# /artifacts (перекрывает запечённые).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# --- Артефакты (model.pt + serving_index/) с HuggingFace Hub ---
# ЗАМЕНИТЕ значение на свой публичный репозиторий с артефактами
# (или переопределите при сборке: --build-arg ARTIFACTS_REPO=user/repo).
ARG ARTIFACTS_REPO=Paulusfmx/avito-visual-search-artifacts
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${ARTIFACTS_REPO}', local_dir='/artifacts')"

# Прогрев кэша базовой SigLIP (через open_clip напрямую — не зависит от ./src,
# чтобы слой не пересобирался при правках кода). Best-effort: при неудаче веса
# скачаются в рантайме.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-16-SigLIP2-256', pretrained='webli')" \
    || echo "base SigLIP cache warm skipped"

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
    IMAGE_URL_MODE=public \
    PUBLIC_BASE_URL=http://89.169.172.66:8000/files

EXPOSE 8000
CMD ["python", "scripts/serve.py"]
