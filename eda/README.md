# EDA & фильтрация Avito women-clothes (1.21M → 1.05M)

Папка содержит результаты фильтрации полного манифеста (1,214,677 строк, 274,138 товаров) на множество релевантных изображений для обучения image-similarity / text-to-image модели.

## Главный артефакт

**[`valid_image_ids.csv`](valid_image_ids.csv)** (или [`valid_image_ids.csv.gz`](valid_image_ids.csv.gz)) — **1,050,324 строки**, формат:

```csv
image_id,item_id
1046006829584,365182752803
1046007064989,365182752803
...
```

- `image_id` — логический ID картинки (стабильный, использовать для джойна с manifest'ами)
- `item_id` — ID товара

### Проверка целостности

```bash
shasum -a 256 valid_image_ids.csv
# d7372d945f0829d8b060b57ec76c7147c4ca1e797b560652a86567aaf536fba4

shasum -a 256 valid_image_ids.csv.gz
# aadf90fb16965d2ebbb06d6614efa66d21010b05e7654dd7fa3faab09974624f
```

### Использование

```python
import pandas as pd

valid = pd.read_csv('valid_image_ids.csv')         # 1,050,324 rows
manifest = pd.read_csv('tmp_manifest_validated.csv')  # 1,214,677 rows

# inner join -> только валидные строки манифеста
train = manifest.merge(valid, on=['image_id', 'item_id'], how='inner')
# train.shape -> (1050324, 37)
```

Или прямо по файлам на диске:

```python
import csv
valid_ids = set()
with open('valid_image_ids.csv') as f:
    next(f)
    for line in f:
        image_id, _ = line.strip().split(',')
        valid_ids.add(image_id)

# image_id из имени файла = image_storage_image_id != image_id (!)
# поэтому без джойна с manifest.csv по двум полям не обойтись
```

## Что было отсеяно и почему

См. подробный отчёт: **[`filtering_report.md`](filtering_report.md)**.

Кратко:

| Этап | Drop | Что |
|---|---:|---|
| `validation_status ≠ 'ok'` | 363 | повреждённые JPEG |
| **`phash_cross_item_dup`** | **163,421** | один phash у > 1 товара = сток-фото / баннеры / служебные плашки. Оставлен 1 представитель на phash |
| `pixel_blank` (entropy<1.5 ∧ edge<1.0) | 16 | пустые / однотонные |
| `pixel_thin` (aspect<0.3 ∨ aspect>3.0) | 553 | узкие полоски-коллажи |
| **Итого:** | **164,353** | **86.5% от исходного 1.21M сохранено** |

Сводка в числах: [`filter_summary.json`](filter_summary.json).

## Воспроизводимость

Скрипты, которыми получен результат:

| Файл | Назначение |
|---|---|
| [`apply_filter.py`](apply_filter.py) | Phase 1 (manifest-only: validation + phash cross-item dedup) + Phase 2 (pixel-blank + pixel-thin). Принимает `tmp_manifest_validated.csv` и опциональный `pixel_stats.csv`. |
| [`run_pixel_stats_full.py`](run_pixel_stats_full.py) | Параллельный расчёт pixel-level статистик (entropy / edge_density / aspect / channel_spread) для всех 1.05M kept-картинок. Используется как вход Phase 2 в `apply_filter.py`. |

Команда полного запуска:

```bash
# Phase 2 требует распакованного датасета: data/full/dataset_1M/images/NNN/MMM/*.jpg
python3 run_pixel_stats_full.py \
  --kept-csv out/relevant_phase1.csv \
  --images-root data/full/dataset_1M/images \
  --workers 12 \
  --out-csv out/pixel_stats_full.csv          # ~3 мин, 5400 img/s на 12 ядрах

python3 apply_filter.py \
  --manifest data/tmp_manifest_validated.csv \
  --pixel-stats out/pixel_stats_full.csv \
  --out-dir out/full                          # ~30 сек
```

## Чего здесь НЕТ (намеренно)

- **`sha256` дубликаты** не выкидывались как класс — это уронило бы 35,659 товаров (13%) полностью; вместо этого через cross-item phash дедуп оставлен 1 представитель на phash.
- **Within-item phash совпадения сохранены** — это разные углы/освещение одной позиции, валидные для аугментаций.
- **CLIP-семантический фильтр** не применялся — на 1M это часы GPU; запланировано как часть дотюнинга CLIP-like модели.
- **Описания / тайтлы товаров отсутствуют в исходном датасете** — `items.csv` содержит только структурированные поля (`predmet_odezhdy`, `cvet`, `brand`, `sostoyanie`). Текстовая сторона text-to-image собирается из них шаблоном.
