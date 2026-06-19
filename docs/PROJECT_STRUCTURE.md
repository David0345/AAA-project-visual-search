# Структура проекта и архитектура кода

## 1. Что мы строим

Мультимодальный поиск по каталогу Avito (женская одежда, ~1M товаров) на базе
CLIP/SigLIP-подобной модели с контрастивным лоссом. Три режима запроса:

- **image** — фото → визуально похожие товары;
- **txt** — текст → товары, подходящие по изображению («футболка с котиком»);
- **multimodal** — фото + текстовый модификатор.

Под капотом единое векторное пространство для картинок и текста: эмбеддинги
каталога считаются офлайн и кладутся в ANN-индекс, на запрос мы кодируем
вход, ищем top-K ближайших и подтягиваем метаданные. Сервис должен отвечать
< 1 сек и держать ≥ 1 rps.

## 2. Принципы раскладки

1. **Весь переиспользуемый код — в одном устанавливаемом пакете `src/visual_search/`.**
   Один и тот же код энкодера используется и в обучении, и в оценке, и в сервисе —
   дублировать его по папкам-скриптам нельзя.
2. **Точки входа (CLI) тонкие.** В `scripts/` лежат запускаемые обёртки на
   несколько строк; вся логика — в пакете. Это держит импорты стабильными и
   позволяет тестировать логику без запуска процессов. (Принцип соблюдён для
   сервиса; исследовательские скрипты в `scripts/` — самостоятельные, см. §9.)
3. **Границы модулей = границы ответственности.** Каждый участник владеет своим
   подпакетом и редко трогает чужие файлы → меньше merge-конфликтов.
4. **Модули общаются через зафиксированные контракты, а не через внутренности
   друг друга** (см. §5). Пока контракт держится, любой может менять реализацию
   у себя.
5. **Данные, чекпойнты и индексы не коммитятся** — лежат в gitignored-папках
   `data/` и `experiments/`. В git только код, конфиги и документация.
6. **Деплой добавляется поверх, а не переписыванием.** `serving/` уже
   зарезервирован и опирается на тот же query pipeline, что и оценка.

## 3. Целевое дерево директорий

```text
AAA-project-visual-search/
├── README.md
├── pyproject.toml                # зависимости + точки входа пакета
├── uv.lock
├── .python-version
│
├── configs/                      # конфиги экспериментов (YAML), без секретов
│   ├── data/                     #   параметры датасета/аугментаций
│   ├── model/                    #   какой энкодер, размерность, head
│   ├── train/                    #   lr, batch, epochs, loss, scheduler
│   └── experiment/               #   композиция data+model+train в один прогон
│
├── src/
│   └── visual_search/            # единый python-пакет (import visual_search...)
│       ├── __init__.py
│       │
│       ├── common/               # общий фундамент, нейтральная зона
│       │   ├── config.py         #   загрузка/валидация конфигов
│       │   ├── io.py             #   пути, чтение parquet/csv, артефакты
│       │   ├── logging.py
│       │   └── seed.py           #   фиксация сидов (воспроизводимость)
│       │
│       ├── data/                 # ┓ ВЛАДЕЛЕЦ: «Подготовка данных»
│       │   ├── dataset.py        # │ torch Dataset поверх train.parquet
│       │   ├── datamodule.py     # │ сборка train/val DataLoader
│       │   ├── collate.py        # │ батчинг (image+text пары)
│       │   ├── transforms.py     # │ аугментации изображений
│       │   ├── tokenization.py   # │ текстовый токенайзер
│       │   └── prepare/          # │ офлайн-подготовка (out-of-loop)
│       │       ├── eda/          # │   ← переезд из ./eda
│       │       └── build_train.py# ┛   ← ./train_dataset/prepare_train_data.py
│       │
│       ├── models/               # ┓ ВЛАДЕЛЕЦ: «Обучение»
│       │   ├── base.py           # │ интерфейс энкодера (контракт §5.2)
│       │   ├── encoders.py       # │ обёртки над open_clip / transformers
│       │   ├── heads.py          # │ проекционные головы
│       │   ├── losses.py         # │ InfoNCE / SigLIP / contrastive
│       │   └── registry.py       # ┛ фабрика «имя из конфига → модель»
│       │
│       ├── training/             # ┓ ВЛАДЕЛЕЦ: «Обучение»
│       │   ├── train.py          # │ оркестрация одного прогона
│       │   ├── loop.py           # │ epoch/step, forward, backward
│       │   ├── optim.py          # │ оптимизатор + scheduler
│       │   ├── checkpoint.py     # │ save/load чекпойнтов
│       │   └── tracking.py       # ┛ логирование метрик (W&B/TB/csv)
│       │
│       ├── evaluation/           # ┓ ВЛАДЕЛЕЦ: «Оценка»
│       │   ├── val_dataset.py    # │ ← ./docs/val_dataset/val_dataset.py
│       │   ├── metrics.py        # │ Recall@k, Precision@k, MRR
│       │   └── evaluate.py       # ┛ прогон val_ds → отчёт по 3 режимам
│       │
│       ├── index/                # общий слой: нужен оценке И сервису
│       │   ├── embed.py          #   батч-инференс эмбеддингов каталога
│       │   ├── ann.py            #   обёртка над faiss/hnswlib
│       │   └── build_index.py    #   эмбеддинги → индекс на диске
│       │
│       └── serving/              # веб-сервис (каркас сейчас, код позже)
│           ├── app.py            #   FastAPI
│           ├── search.py         #   query pipeline: encode→ANN→hydrate
│           └── schemas.py        #   request/response модели
│
├── scripts/                      # тонкие CLI-обёртки над пакетом
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── build_index.py
│   └── serve.py
│
├── notebooks/                    # разведочные ноутбуки (не для продакшна)
│
├── experiments/                  # АРТЕФАКТЫ прогонов — gitignored
│   └── <run_id>/                 #   checkpoints/ logs/ metrics.json config.yaml
│
├── data/                         # ДАННЫЕ — gitignored
│   ├── raw/                      #   исходный dataset_1M (images + csv)
│   ├── interim/                  #   valid_image_ids.csv и пр.
│   └── processed/                #   train.parquet, эмбеддинги, индексы
│
├── docs/                         # документация (этот файл, оценка, val_dataset)
└── tests/                        # pytest, по подпакету на каталог
```

## 4. Карта ответственности

Срез train-loop — **Подготовка данных / Обучение / Оценка** — ложится на модули так:

| Участник (роль)        | Владеет модулями                        | Главный артефакт на выходе                    |
|------------------------|-----------------------------------------|-----------------------------------------------|
| **Подготовка данных**  | `data/` (+ `data/prepare/`)             | `train.parquet`, `DataLoader`'ы               |
| **Обучение**           | `models/` + `training/`                 | обученный чекпойнт в `experiments/<run_id>/`  |
| **Оценка**             | `evaluation/` (+ участие в `index/`)    | отчёт с Recall@10 / Precision@10 / MRR        |
| **Общее**              | `common/`, `index/`, `serving/`, `configs/`, `scripts/` | меняются по согласованию (PR-review) |

Существующие папки переезжают в пакет (содержимое сохраняем как есть):

| Сейчас                          | Переезжает в                              |
|---------------------------------|-------------------------------------------|
| `eda/`                          | `src/visual_search/data/prepare/eda/`     |
| `train_dataset/`                | `src/visual_search/data/prepare/`         |
| `docs/val_dataset/*.py`         | `src/visual_search/evaluation/`           |
| `docs/val_dataset/*.csv, *.md`  | остаются в `docs/val_dataset/` (это данные/доки) |

> Переезд можно делать постепенно: новый код сразу пишем в `src/`, старые папки
> переносим отдельными PR, чтобы не ломать историю. Главное — не плодить
> параллельные копии одной логики.

## 5. Контракты между модулями

Это самая важная часть: пока контракты держатся, три человека работают
независимо. Любое изменение контракта — через обсуждение и обновление этого
раздела.

### 5.1. Контракт «Подготовка данных → Обучение»: схема `train.parquet`

Фиксируется как есть (см. `data/prepare/` README). Тренировочный код опирается
**только** на эти колонки:

| Колонка             | Тип        | Назначение                                  |
|---------------------|------------|---------------------------------------------|
| `item_id`           | int        | ID товара                                   |
| `title_image_path`  | str        | путь к титульному изображению               |
| `other_image_paths` | list[str]  | прочие изображения товара                   |
| `product_text`      | str        | структурированное описание                  |
| `queries`           | list[str]  | сгенерированные текстовые запросы           |
| `predmet_odezhdy`, `cvet`, `brand`, `param2`, `sostoyanie`, `category_name` | str | метаданные (фильтры/негативы) |

Пути к изображениям — относительные от `data/raw/dataset_1M/`.

### 5.2. Контракт «модель» (ядро всей системы)

Любая модель в `models/` реализует единый интерфейс — на него опираются
обучение, оценка и сервис:

```python
class Encoder(Protocol):
    embed_dim: int
    def encode_image(self, images: Tensor) -> Tensor: ...   # -> (B, embed_dim), L2-norm
    def encode_text(self, tokens: Tensor) -> Tensor:  ...   # -> (B, embed_dim), L2-norm
```

- эмбеддинги **L2-нормированы**, схожесть = косинус = скалярное произведение;
- `embed_dim` объявлен и неизменен в рамках одного индекса;
- создание — только через `models.registry.build_model(config)` (никаких прямых
  импортов конкретного класса из тренировочного кода).

### 5.3. Контракт «Оценка»: схема `val_dataset` и метрики

Колонки val_ds (`query_id, mode, item_id, image_id, image_path, txt_query,
target_images_id, ...`) фиксированы (см. `docs/val_dataset/VAL_DATASET.md`).
`evaluation.evaluate(model, index)` возвращает по каждому режиму
(`image/txt/multimodal`) словарь: `{"recall@10", "precision@10", "mrr"}` —
micro и macro по категориям.

### 5.4. Контракт «эмбеддинги/индекс»: Оценка ↔ Сервис

`index/` принимает каталожные эмбеддинги `(N, embed_dim) float32, L2-norm` +
параллельный массив `item_id/image_id` и строит ANN-индекс. Поиск:
`ann.search(query_vec, k) -> [(image_id, score), ...]`. Один и тот же индекс
используется в офлайн-оценке и в онлайн-сервисе — это гарантирует, что метрики
соответствуют поведению прода.

## 6. Поток данных (end-to-end)

```text
            ОФЛАЙН (out-of-loop)                        ОНЛАЙН (сервис)
                                                         запрос (img/txt)
 raw images+csv                                                │
      │  data/prepare (eda → фильтр)                  serving/search.py
      ▼                                                        │ encode_*
 valid_image_ids.csv                                           ▼
      │  data/prepare/build_train.py                  index/ann.search(k)
      ▼                                                        │
 train.parquet ──► data/DataLoader ──► training/loop           ▼
      │                                   │ checkpoint   hydrate item_id→
      │                                   ▼              (URL, заголовок)
      │                          experiments/<run_id>           │
      │                                   │                     ▼
      └──► index/embed (каталог) ◄────────┘              top-N галерея
                   │
                   ▼
            index/build_index ──► ANN-индекс (используется оценкой и сервисом)
                   │
                   ▼
            evaluation/evaluate ──► Recall@10 / Precision@10 / MRR
```

## 7. Конфиги и эксперименты

- Каждый прогон полностью описывается конфигом из `configs/experiment/`
  (композиция `data + model + train`). Никаких «магических» значений в коде.
- Прогон пишет всё в `experiments/<run_id>/`: `config.yaml` (копия),
  `checkpoints/`, `logs/`, `metrics.json`. По `run_id` эксперимент
  воспроизводим: тот же конфиг + тот же сид → тот же результат.
- `experiments/` и `data/` — в `.gitignore`. Делиться весами/индексами — через
  внешнее хранилище (S3/диск), не через git.

## 8. Команды (целевой интерфейс CLI)

```bash
# 1. подготовка данных  (владелец: Подготовка данных)
python scripts/prepare_data.py --config configs/data/baseline.yaml

# 2. обучение           (владелец: Обучение)
python scripts/train.py --config configs/experiment/clip_baseline.yaml

# 3. индекс каталога    (общее)
python scripts/build_index.py --checkpoint experiments/<run_id>/checkpoints/best.pt

# 4. оценка             (владелец: Оценка)
python scripts/evaluate.py --checkpoint experiments/<run_id>/... --index ...

# 5. сервис             (позже, для деплоя)
python scripts/serve.py --checkpoint ... --index ...
```

## 9. Фактическое состояние: реальный пайплайн vs целевой каркас

Разделы 3–8 описывают **целевую** раскладку. По факту в репозитории сосуществуют
два пути обучения, и при чтении кода это важно учитывать.

**Каркас (reference).** Пакетный пайплайн — `training/` (`train.py`, `loop.py`,
`optim.py`, `checkpoint.py`, `tracking.py`), тонкие
`scripts/{train,prepare_data,build_index,evaluate}.py`,
`data/{dataset,datamodule,collate}.py`, `evaluation/evaluate.py`,
`index/{build_index,embed}.py`. Это чистая Hydra-архитектура «как должно быть».
Финальную модель она **не** обучала и под open_clip/SigLIP в текущем виде не
настроена (`data/dataset.py` рассчитан на `transformers.CLIPProcessor`). Оставлен
как референс структуры и контрактов.

**Фактический пайплайн экспериментов.** Победитель получен так:
- запросы: `scripts/gen_queries_qwen.py` (Qwen2-VL по фото) →
  `scripts/build_synth_train.py` → `train_synth.parquet`;
- обучение: `scripts/finetune_mini.py` — единый тренер всех прогонов из
  `experiments/metrics_ledger.jsonl` (open_clip preprocess/tokenizer, InfoNCE/
  Sigmoid, AdamW + warmup/cosine, AMP bf16, опц. hard-negative батчинг);
- оценка: `scripts/eval_full.py` — каталог 52k (таргеты + 50k дистракторов) →
  `experiments/eval_full_ledger.jsonl`;
- независимый held-out: `scripts/build_gemini_eval.py`;
- индекс и сервис: `scripts/build_catalog_index.py` →
  `scripts/repack_index_for_serving.py` → `serving/`.

**Скрипты.** Принцип §2.2 (тонкие обёртки) выполнен только для `serve.py`;
остальные ~25 скриптов в `scripts/` — исследовательские (свипы, генерация данных,
оценка). Это артефакты экспериментов, хранятся намеренно.

**Расхождения с деревом §3.** Целевые, пока не реализованы: `common/config.py`,
`data/tokenization.py`, `models/heads.py` (заглушка). Реально присутствуют, но не
указаны в §3: `serving/index.html` (фронтенд), `index/benchmark.py` (выбор
пулинга по recall@10), синтетика и `eval_full.py`.

**Воспроизводимость синтетики и eval.** `gen_queries_qwen.py` по умолчанию
генерит **v1 (descriptive)** — синтетику финальной модели; `--style multi` даёт
v2 (5 стилей). `eval_full.py` берёт каталог из `--images-csv` (по умолчанию
`data/raw/dataset_1M/images.csv`) и `--valid-ids`, абсолютных путей нет.
`finetune_mini.py` пишет в ledger реальное имя модели (`--model`).
