"""Сборка ANN-индекса из эмбеддингов каталога и сохранение на диск.

Это логика; запускается тонкой обёрткой scripts/build_index.py. Результат —
в data/processed/ (индекс + маппинг id), переиспользуется оценкой и сервисом.

TODO(index): build_index(checkpoint, catalog) -> сохранённый индекс.
"""

from __future__ import annotations
