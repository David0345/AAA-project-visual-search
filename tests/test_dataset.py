import pandas as pd
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import patch
import pytest

from visual_search.data.dataset import ContrastiveImageTextDataset, SearchEvalDataset


# Фейковый процессор, который имитирует интерфейс без загрузки модели
class FakeProcessor:
    """Минимальная заглушка процессора для тестов."""
    def __call__(self, text=None, images=None, return_tensors='pt',
                 padding='max_length', truncation=True, **kwargs):
        result = {}
        if images is not None:
            # Возвращаем фейковые pixel_values [B, 3, 224, 224]
            n = len(images) if isinstance(images, list) else 1
            result['pixel_values'] = torch.randn(n, 3, 224, 224)
        if text is not None:
            # Возвращаем фейковые input_ids [B, 77]
            n = len(text) if isinstance(text, list) else 1
            result['input_ids'] = torch.randint(0, 1000, (n, 77))
            result['attention_mask'] = torch.ones(n, 77)
        return result


@pytest.fixture()
def fake_processor():
    return FakeProcessor()


@pytest.fixture()
def fake_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        'item_id': [1, 2],
        'title_image_path': ['img1.jpg', 'img2.jpg'],
        'queries': [['тест1', 'запрос1'], ['тест2']]
    })
    p = tmp_path / 'fake_train.parquet'
    df.to_parquet(p)
    return p


def test_contrastive_dataset_getitem(fake_parquet, tmp_path, fake_processor):
    ds = ContrastiveImageTextDataset(
        parquet_path=str(fake_parquet),
        image_root=str(tmp_path),
        processor=fake_processor,
        seed=42
    )

    fake_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    with patch('PIL.Image.open', return_value=fake_img):
        item = ds[0]

    assert item['image'].shape == (3, 224, 224)
    assert item['input_ids'].dim() == 1
    assert item['item_id'] == 1
    assert isinstance(item['text_raw'], str)


def test_search_eval_dataset_modes(fake_parquet, tmp_path, fake_processor):
    eval_csv = tmp_path / 'val.csv'
    pd.DataFrame({
        'query_id': [10, 20],
        'mode': ['txt', 'image'],
        'item_id': [1, 2],
        'image_path': ['img1.jpg', 'img2.jpg'],
        'txt_query': ['чёрное платье', None],
        'target_images_id': ['1001; 1002', '2001'],
        'param2': ['Платья', 'Верхняя одежда'],
        'brand': ['Zara', 'H&M'],
        'cvet': ['Чёрный', 'Синий'],
        'category_name': ['Женская одежда', 'Женская одежда']
    }).to_csv(eval_csv, index=False)

    ds = SearchEvalDataset(
        csv_path=str(eval_csv),
        image_root=str(tmp_path),
        processor=fake_processor
    )

    fake_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    with patch('PIL.Image.open', return_value=fake_img):
        q_txt = ds[0]
        q_img = ds[1]

    assert q_txt['mode'] == 'txt'
    assert q_txt['query']['pixel_values'] is None
    assert q_txt['query']['input_ids'] is not None
    assert q_txt['target_ids'] == [1001, 1002]

    assert q_img['mode'] == 'image'
    assert q_img['query']['pixel_values'] is not None
    assert q_img['query']['input_ids'] is None
