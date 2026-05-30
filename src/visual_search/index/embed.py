"""Батч-инференс эмбеддингов каталога обученной моделью.

Выход: (N, embed_dim) float32 L2-norm + параллельный массив image_id/item_id.

TODO(index): embed_catalog(model, dataloader) -> (vectors, ids).
"""

from __future__ import annotations
