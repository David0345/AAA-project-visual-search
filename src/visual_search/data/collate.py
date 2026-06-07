"""Collate-функции для сборки батчей из датасетов."""

from __future__ import annotations

import torch
from typing import List, Dict, Any


def contrastive_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Собирает батч для контрастивного обучения.

    Вход: список словарей с ключами:
        - image: Tensor [3, 224, 224]
        - input_ids: Tensor [seq_len]
        - attention_mask: Tensor [seq_len] (опционально)
        - item_id: int

    Выход:
        {
            'images': Tensor [B, 3, 224, 224],
            'input_ids': Tensor [B, seq_len],
            'attention_mask': Tensor [B, seq_len],
            'item_ids': List[int]
        }
    """
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])

    if 'attention_mask' in batch[0] and batch[0]['attention_mask'] is not None:
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
    else:
        attention_mask = torch.ones_like(input_ids)

    item_ids = [item['item_id'] for item in batch]

    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'item_ids': item_ids,
    }


def eval_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Для eval-датасета возвращаем список, т.к. запросы разной модальности.

    Каждый элемент батча - готовый к инференсу запрос.
    """
    return batch  # batch_size=1 в eval, поэтому просто возвращаем как есть
