# Handoff: запуск экспериментов на A800

## Контекст

Проект — мультимодальный визуальный поиск по каталогу Авито (женская одежда, ~1M товаров).
Три режима: `image` (фото→похожие), `txt` (русский текст→фото), `multimodal` (фото+текст-модификатор).
Метрики: Recall@K, Precision@K, MRR — агрегированные micro-avg + разбивка по категориям (param2).

Базовая модель: `xlm-roberta-base-ViT-B-32` (open_clip, pretrained=laion5b_s13b_b90k) — знает русский.
Библиотека: `open_clip`, НЕ HuggingFace transformers.

## Что уже сделано локально

### Zero-shot baseline (xlm_clip_vit_b32, без обучения)
```
Mode          N     R@10    MRR
image        175    0.001   0.001   ← артефакт маленького каталога (477 img)
txt          185    0.282   0.602   ← сильный: xlm-roberta знает русский
multimodal   182    0.072   0.062
all          542    0.121   0.227
```
Результаты: `experiments/zeroshot/xlm_clip_vit_b32/metrics.json`

### Smoke-тест файн-тюнинга (300 шагов, MPS, batch=32)
```
Mode          MRR baseline → after 300 steps   Δ
txt           0.602 → 0.571                    -0.031  (нормально для старта)
multimodal    0.062 → 0.137                    +0.075  (+121%)
all           0.227 → 0.242                    +0.015
```
Скорость на MPS: ~23 img/s → ~30 мин/эпоха при 40k товарах.
На A800 с batch=256: ожидаем ~2000+ img/s → **~1-2 мин/эпоха** на 40k, ~30 мин на 1M.

## Структура данных

```
data/
  raw/
    dataset_1M/
      images/          ← 50k jpg (sample) или полный 1M после распаковки архива
      images.csv       ← метаданные изображений (image_id, item_id, image_path, is_title)
      tmp_manifest_with_urls.csv  ← метаданные товаров (predmet_odezhdy, cvet, brand, ...)
  interim/
    mini_train.parquet ← уже подготовленный датасет: 40,597 товаров, avg 43.5 запросов/товар
src/
  visual_search/
    data/eda/
      valid_image_ids.csv  ← EDA-фильтр: 1,050,324 валидных (image_id, item_id)
    evaluation/
      val_dataset/
        val_dataset.csv  ← 542 валидационных запроса (image/txt/multimodal)
```

## Установка окружения

```bash
cd ~/personal/AAA-project-visual-search

# Если есть uv:
uv sync
source .venv/bin/activate

# Или pip:
pip install -e ".[dev]"
pip install wandb  # если нужен логгинг
```

На A800 torch уже установлен с CUDA. Если нет — `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

## Подготовка полного train.parquet (если есть полный архив)

Если перенесли `upload.tar.gz` (39 GB) — сначала распаковать:
```bash
cd ~/personal
tar -xzf upload.tar.gz -C AAA-project-visual-search/data/raw/
# Получится: data/raw/dataset_1M/images/ (1M картинок)
```

Затем собрать полный train.parquet:
```bash
cd ~/personal/AAA-project-visual-search
python scripts/prepare_mini_train.py \
    --data-dir data/raw/dataset_1M \
    --valid-ids src/visual_search/data/eda/valid_image_ids.csv \
    --output data/interim/train_full.parquet
# Ожидаем: ~900k товаров (после EDA-фильтра)
```

## Запуск экспериментов

### 1. Zero-shot baseline (проверка что всё работает)
```bash
python scripts/zeroshot_eval.py \
    --model xlm_clip_vit_b32 \
    --device cuda \
    --images-base data/raw/dataset_1M
# Результат: experiments/zeroshot/xlm_clip_vit_b32/metrics.json
```

### 2. Smoke-тест файн-тюнинга (mini_train.parquet, ~5 мин на A800)
```bash
python scripts/finetune_mini.py \
    --device cuda \
    --batch-size 256 \
    --epochs 3 \
    --lr 5e-6 \
    --images-base data/raw/dataset_1M \
    --train-parquet data/interim/mini_train.parquet \
    --out-dir experiments/finetune_mini_a800
```

### 3. Полное обучение (train_full.parquet, если готов)
```bash
python scripts/finetune_mini.py \
    --device cuda \
    --batch-size 512 \
    --epochs 5 \
    --lr 5e-6 \
    --num-workers 8 \
    --images-base data/raw/dataset_1M \
    --train-parquet data/interim/train_full.parquet \
    --out-dir experiments/finetune_full
```

### 4. Через Hydra (более гибко, с конфигами)
```bash
python scripts/train.py \
    experiment=xlm_clip_baseline \
    data.train_path=data/interim/train_full.parquet \
    data.image_root=data/raw/dataset_1M \
    data.batch_size=512 \
    train.epochs=5 \
    train.lr=5e-6
```

## Ключевые параметры для A800

| Параметр | MPS (local) | A800 (рекомендация) |
|----------|-------------|---------------------|
| batch_size | 32 | 256–512 |
| lr | 1e-5 | 5e-6 (full), 1e-5 (frozen backbone) |
| epochs | 1-2 | 3-5 |
| num_workers | 0 | 8 |
| amp | нет | да (--amp или train.amp=true) |

## Архитектура моделей

```python
# src/visual_search/models/encoders.py
_MODEL_MAP = {
    "clip_vit_b32":     ("ViT-B-32", "openai"),           # EN
    "clip_vit_b16":     ("ViT-B-16", "openai"),           # EN
    "clip_vit_l14":     ("ViT-L-14", "openai"),           # EN, embed=768
    "xlm_clip_vit_b32": ("xlm-roberta-base-ViT-B-32",     # RU! основная
                          "laion5b_s13b_b90k"),
}
```

## Эксперименты для подбора гиперпараметров

Предлагаемый порядок:

1. **xlm_clip_baseline** — полное обучение, lr=5e-6, batch=512, 3 эпохи
2. **xlm_clip_frozen_backbone** — заморозить visual+text, обучать только logit_scale (быстро, проверка)
3. **xlm_clip_lr_sweep** — sweep по lr: [1e-6, 5e-6, 1e-5]
4. **clip_sigmoid_loss** — SigmoidLoss вместо InfoNCE (лучше для маленьких батчей)

Конфиги: `configs/experiment/`

## Логирование

Метрики пишутся в `experiments/<run_name>/run_log.json`.
Для wandb добавить `--wandb` или в конфиге `wandb.enabled=true`.

После каждого запуска eval-результаты сравниваются с zero-shot baseline автоматически.

## Известные проблемы

- `OMP Error #15 (duplicate libomp)` на Mac: `export KMP_DUPLICATE_LIB_OK=TRUE`. На Linux не нужно.
- `faiss-cpu` vs `faiss-gpu`: для индексирования 1M векторов лучше `pip install faiss-gpu`.
- `num_workers > 0` с MPS → краши. На CUDA работает нормально.
- `image` mode метрики низкие на val (~0.001) — артефакт маленького val-каталога (477 img), не баг модели.

## Структура кода (кратко)

```
src/visual_search/
  models/
    encoders.py      ← _OpenCLIPEncoder, регистрация через @register()
    losses.py        ← InfoNCELoss, SigmoidLoss
    registry.py      ← build_model(config), get_processor()
  data/
    dataset.py       ← ContrastiveImageTextDataset (train), SearchEvalDataset (val)
    datamodule.py    ← build_dataloaders(config)
    prepare/
      build_train.py       ← генерация train.parquet из items.csv + images.csv
      category_synonyms.py ← синонимы категорий для запросов
      brand_translit.py    ← русские названия брендов
  evaluation/
    metrics.py       ← recall_at_k, precision_at_k, mrr, aggregate()
    val_dataset.py   ← ValDataset (читает val_dataset.csv)
    evaluate.py      ← evaluate(), print_report()
  index/
    ann.py           ← ANNIndex (flat/ivf/ivfpq/hnsw через FAISS)
  training/
    train.py         ← основной train loop (Hydra)
    tracking.py      ← MetricsTracker (JSON + WandB)

scripts/
  zeroshot_eval.py   ← zero-shot оценка без обучения
  finetune_mini.py   ← standalone файн-тюнинг (без Hydra, проще)
  prepare_mini_train.py ← сборка mini_train.parquet из local images
  train.py           ← Hydra-based полное обучение
```
