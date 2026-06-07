"""Тесты модуля оценки: metrics, SearchEvalDataset, evaluate.
Работают на чистом Python + минимальные моки PyTorch.
Запуск: pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import patch
from PIL import Image


from visual_search.evaluation.metrics import (
    ModeMetrics,
    aggregate,
    mrr,
    precision_at_k,
    recall_at_k,
)
from visual_search.data.dataset import SearchEvalDataset
from visual_search.evaluation.evaluate import evaluate_with_search_fn


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_all_relevant_in_top(self):
        assert recall_at_k([1, 2, 3, 4], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_partial_recall(self):
        # 1 из 3 таргетов в топ-2
        assert recall_at_k([1, 99, 2, 3], {1, 2, 3}, k=2) == pytest.approx(1 / 3)

    def test_nothing_found(self):
        assert recall_at_k([10, 20], {1, 2, 3}, k=2) == pytest.approx(0.0)

    def test_empty_targets(self):
        assert recall_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)

    def test_k_larger_than_list(self):
        # k > len(ranked) — нет ошибки, просто усечение
        assert recall_at_k([1, 2], {1, 2, 3}, k=100) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_half_precision(self):
        assert precision_at_k([1, 99, 2, 88], {1, 2}, k=4) == pytest.approx(2 / 4)

    def test_zero_precision(self):
        assert precision_at_k([10, 20], {1, 2}, k=2) == pytest.approx(0.0)

    def test_empty_targets(self):
        assert precision_at_k([1, 2], set(), k=2) == pytest.approx(0.0)

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1, 2}, k=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_result_relevant(self):
        assert mrr([1, 2, 3], {1}) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mrr([99, 1, 2], {1}) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        assert mrr([99, 88, 1], {1}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr([10, 20, 30], {1, 2}) == pytest.approx(0.0)

    def test_multiple_targets_first_found(self):
        # таргеты {2, 3}, первый найденный на позиции 2 → MRR = 0.5
        assert mrr([99, 2, 3], {2, 3}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_basic(self):
        ranks = [[1, 2, 3], [4, 5, 6]]
        targets = [{1}, {4}]
        m = aggregate(ranks, targets, k_values=[1, 3], mode="test")
        assert m.mode == "test"
        assert m.count == 2
        assert m.recall_at_k[1] == pytest.approx(1.0)   # оба первых релевантны
        assert m.precision_at_k[1] == pytest.approx(1.0)
        assert m.mrr_score == pytest.approx(1.0)

    def test_skips_empty_targets(self):
        ranks = [[1], [2], [3]]
        targets = [{1}, set(), {3}]   # второй запрос без ответа
        m = aggregate(ranks, targets, k_values=[1])
        assert m.count == 2  # пустой таргет пропущен

    def test_per_category_breakdown(self):
        ranks = [[1], [2], [3]]
        targets = [{1}, {2}, {99}]
        cats = ["Юбки", "Юбки", "Платья"]
        m = aggregate(ranks, targets, k_values=[1], per_query_categories=cats)
        assert "Юбки" in m.per_category
        assert "Платья" in m.per_category
        assert m.per_category["Юбки"]["r@1"] == pytest.approx(1.0)
        assert m.per_category["Платья"]["r@1"] == pytest.approx(0.0)

    def test_as_flat_dict(self):
        m = ModeMetrics(
            mode="txt", count=10,
            recall_at_k={10: 0.75}, precision_at_k={10: 0.30}, mrr_score=0.61,
        )
        d = m.as_flat_dict()
        assert d["mode"] == "txt"
        assert d["recall@10"] == pytest.approx(0.75)
        assert d["precision@10"] == pytest.approx(0.30)
        assert d["mrr"] == pytest.approx(0.61)


# ---------------------------------------------------------------------------
# Тесты SearchEvalDataset
# ---------------------------------------------------------------------------
class FakeProcessor:
    """Минимальная заглушка процессора для тестов."""
    def __call__(self, text=None, images=None, return_tensors='pt', padding='max_length', truncation=True, **kwargs):
        result = {}
        if images is not None:
            n = len(images) if isinstance(images, list) else 1
            result['pixel_values'] = torch.randn(n, 3, 224, 224)
        if text is not None:
            n = len(text) if isinstance(text, list) else 1
            result['input_ids'] = torch.randint(0, 1000, (n, 77))
            result['attention_mask'] = torch.ones(n, 77)
        return result


class TestSearchEvalDataset:
    @pytest.fixture
    def fake_eval_csv(self, tmp_path: Path) -> Path:
        eval_csv = tmp_path / 'val.csv'
        pd.DataFrame({
            'query_id': [10, 20, 30],
            'mode': ['txt', 'image', 'multimodal'],
            'item_id': [1, 2, 3],
            'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'txt_query': ['чёрное платье', None, 'платье красное'],
            'target_images_id': ['1001; 1002', '2001', '3001; 3002'],
            'param2': ['Платья', 'Верхняя одежда', 'Платья'],
            'category_name': ['Женская одежда', 'Женская одежда', 'Женская одежда'],
            'brand': ['Zara', 'H&M', 'Mango'],
            'cvet': ['Чёрный', 'Синий', 'Красный']
        }).to_csv(eval_csv, index=False)
        return eval_csv

    def test_dataset_length_and_modes(self, fake_eval_csv, tmp_path):
        ds = SearchEvalDataset(
            csv_path=str(fake_eval_csv),
            image_root=str(tmp_path),
            processor=FakeProcessor()
        )
        assert len(ds) == 3
        
        fake_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        with patch('PIL.Image.open', return_value=fake_img):
            q_txt = ds[0]
            q_img = ds[1]
            q_multi = ds[2]

        # Проверка TXT
        assert q_txt['mode'] == 'txt'
        assert q_txt['query']['pixel_values'] is None
        assert q_txt['query']['input_ids'] is not None
        assert q_txt['query']['attention_mask'] is not None
        assert q_txt['target_ids'] == [1001, 1002]

        # Проверка IMAGE
        assert q_img['mode'] == 'image'
        assert q_img['query']['pixel_values'] is not None
        assert q_img['query']['input_ids'] is None
        assert q_img['target_ids'] == [2001]

        # Проверка MULTIMODAL
        assert q_multi['mode'] == 'multimodal'
        assert q_multi['query']['pixel_values'] is not None
        assert q_multi['query']['input_ids'] is not None
        assert q_multi['query'].get('multimodal_alpha') == 0.5


# ---------------------------------------------------------------------------
# Тесты evaluate_with_search_fn
# ---------------------------------------------------------------------------
class MockEvalDataset:
    """Минимальный мок-датасет, имитирующий вывод SearchEvalDataset + eval_collate_fn"""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


class TestEvaluateWithSearchFn:
    def test_evaluate_with_mock_search(self):
        # Создаем мок-элементы, идентичные выводу SearchEvalDataset
        mock_items = [
            {
                'mode': 'txt',
                'target_ids': [1001, 1002],
                'query': {'input_ids': torch.tensor([1, 2]), 'pixel_values': None},
                'metadata': {'category_name': 'Платья'}
            },
            {
                'mode': 'image',
                'target_ids': [2001],
                'query': {'input_ids': None, 'pixel_values': torch.randn(3, 224, 224)},
                'metadata': {'category_name': 'Брюки'}
            }
        ]
        dataset = MockEvalDataset(mock_items)

        # Поиск всегда возвращает пустой список → все метрики 0
        results = evaluate_with_search_fn(
            search_fn=lambda item: [],
            dataset=dataset,
            k_values=[10],
            verbose=False,
        )
        
        assert "all" in results
        assert results["all"].recall_at_k[10] == pytest.approx(0.0)
        assert results["all"].mrr_score == pytest.approx(0.0)
        assert results["all"].count == 2

    def test_evaluate_perfect_search(self):
        mock_items = [
            {
                'mode': 'txt',
                'target_ids': [1001, 1002],
                'query': {'input_ids': torch.tensor([1, 2]), 'pixel_values': None},
                'metadata': {'category_name': 'Платья'}
            }
        ]
        dataset = MockEvalDataset(mock_items)

        def perfect_search(item):
            # Возвращаем все таргеты на первых позициях
            return list(item['target_ids']) + [999999]

        results = evaluate_with_search_fn(
            search_fn=perfect_search,
            dataset=dataset,
            k_values=[10],
            verbose=False,
        )
        
        assert results["all"].recall_at_k[10] == pytest.approx(1.0)
        assert results["all"].mrr_score == pytest.approx(1.0)