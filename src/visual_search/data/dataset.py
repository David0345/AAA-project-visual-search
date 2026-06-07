"""
torch Dataset поверх train.parquet
Отдаёт пары (изображение, текст) для контрастивного обучения. Читает только
зафиксированные колонки train.parquet; пути к картинкам — относительные от
data/raw/dataset_1M/.
"""
from __future__ import annotations

import logging
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor

log = logging.getLogger(__name__)


class ContrastiveImageTextDataset(Dataset):
    """Датасет для контрастивного pretraining RuCLIP.

    Возвращает пары (изображение, текст) из train.parquet.
    Для каждого товара случайно сэмплирует один запрос из списка `queries`.
    """

    def __init__(
        self,
        parquet_path: str,
        image_root: str,
        processor: CLIPProcessor,
        seed: int = 42,
        max_queries_per_item: Optional[int] = None,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.image_root = Path(image_root)
        self.processor = processor
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.max_queries = max_queries_per_item
        self.indices = list(range(len(self.df)))

        if 'queries' in self.df.columns and self.df['queries'].dtype == object:
            self.df['queries'] = self.df['queries'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        log.info(f'Loaded ContrastiveDataset: {len(self.df)} items from {parquet_path}')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = self.image_root / row['title_image_path']
        image = Image.open(img_path).convert('RGB')

        queries = row['queries']
        if self.max_queries and len(queries) > self.max_queries:
            queries = queries[:self.max_queries]

        q_idx = int(torch.randint(len(queries), (1,), generator=self.rng).item())
        text = queries[q_idx]

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        return {
            'image': inputs['pixel_values'].squeeze(0),      # [3, 224, 224]
            'input_ids': inputs['input_ids'].squeeze(0),      # [seq_len]
            'attention_mask': inputs['attention_mask'].squeeze(0),  # [seq_len]
            'item_id': int(row['item_id']),
            'text_raw': text,  # для дебага / логирования
        }


class SearchEvalDataset(Dataset):
    """Датасет для offline-оценки поиска (val/test).

    Поддерживает три режима:
    - image: запрос = изображение, поиск визуально похожих
    - txt: запрос = текст, поиск по семантике
    - multimodal: запрос = изображение + текст-модификатор

    Возвращает готовый к инференсу запрос + ground-truth targets.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        processor: CLIPProcessor,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.processor = processor
        log.info(f'Loaded SearchEvalDataset: {len(self.df)} queries from {csv_path}')

    def __len__(self) -> int:
        return len(self.df)

    def _parse_target_ids(self, target_str: str) -> List[int]:
        """Парсит строку target_images_id в список int.

        Поддерживает форматы:
            "{1045112250624, 1045109001531}"  — Python set repr (основной)
            "123;456;789"                     — устаревший разделитель «;»
        """
        if pd.isna(target_str) or not str(target_str).strip():
            return []
        raw = str(target_str).strip()
        if raw.startswith("{") and raw.endswith("}"):
            return [int(x.strip()) for x in raw[1:-1].split(",") if x.strip()]
        sep = ";" if ";" in raw else ","
        return [int(x.strip()) for x in raw.split(sep) if x.strip()]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        mode = row['mode']
        target_ids = self._parse_target_ids(row.get('target_images_id', ''))

        query = {'pixel_values': None, 'input_ids': None, 'attention_mask': None}

        if mode == 'txt':
            text = str(row['txt_query'])
            inputs = self.processor(text=[text], return_tensors='pt', padding='max_length', truncation=True)
            query['input_ids'] = inputs['input_ids'].squeeze(0)
            query['attention_mask'] = inputs['attention_mask'].squeeze(0)

        elif mode == 'image':
            img_path = self.image_root / row['image_path']
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=[image], return_tensors='pt')
            query['pixel_values'] = inputs['pixel_values'].squeeze(0)

        elif mode == 'multimodal':
            # Стратегия (может поменяем): энкодим раздельно, суммируем эмбеддинги с весом α
            img_path = self.image_root / row['image_path']
            image = Image.open(img_path).convert('RGB')
            text = str(row['txt_query'])

            img_inputs = self.processor(images=[image], return_tensors='pt')
            txt_inputs = self.processor(text=[text], return_tensors='pt', padding='max_length', truncation=True)

            query['pixel_values'] = img_inputs['pixel_values'].squeeze(0)
            query['input_ids'] = txt_inputs['input_ids'].squeeze(0)
            query['attention_mask'] = txt_inputs['attention_mask'].squeeze(0)
            query['multimodal_alpha'] = 0.5  # вес для text-эмбеддинга (можно сделать настраиваемым)

        else:
            raise ValueError(f'Unknown query mode: {mode}')

        return {
            'query': query,
            'target_ids': target_ids,
            'query_id': int(row['query_id']),
            'mode': mode,
            'metadata': {
                'param2': str(row['param2']) if pd.notna(row['param2']) else None,
                'brand': str(row['brand']) if pd.notna(row['brand']) else None,
                'cvet': str(row['cvet']) if pd.notna(row['cvet']) else None,
                'category_name': str(row['category_name']) if pd.notna(row['category_name']) else None,
            }
        }
