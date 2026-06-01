"""Тесты модуля оценки: metrics, val_dataset, evaluate.

Не требуют torch, модели или индекса — работают на чистом Python.
Запуск: pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import pytest

from visual_search.evaluation.metrics import (
    ModeMetrics,
    aggregate,
    mrr,
    precision_at_k,
    recall_at_k,
)
from visual_search.evaluation.val_dataset import _parse_target_ids


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
# _parse_target_ids
# ---------------------------------------------------------------------------


class TestParseTargetIds:
    def test_set_repr(self):
        assert _parse_target_ids("{1, 2, 3}") == {1, 2, 3}

    def test_single_id(self):
        assert _parse_target_ids("{42}") == {42}

    def test_semicolon_separated(self):
        assert _parse_target_ids("1;2;3") == {1, 2, 3}

    def test_comma_separated(self):
        assert _parse_target_ids("1,2,3") == {1, 2, 3}

    def test_empty_string(self):
        assert _parse_target_ids("") == set()

    def test_large_ids(self):
        s = "{1045112250624, 1045109001531, 1045112753893}"
        result = _parse_target_ids(s)
        assert result == {1045112250624, 1045109001531, 1045112753893}


# ---------------------------------------------------------------------------
# ValDataset (требует pandas — пропускаем, если не установлен)
# ---------------------------------------------------------------------------


try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


@pytest.mark.skipif(not _PANDAS_AVAILABLE, reason="pandas не установлен")
class TestValDataset:
    """Интеграционный тест — читает реальный val_dataset.csv."""

    def test_loads_and_stats(self):
        from visual_search.evaluation.val_dataset import ValDataset
        ds = ValDataset()
        st = ds.stats()
        assert st["total"] > 0
        assert st["image"] + st["txt"] + st["multimodal"] == st["total"]

    def test_getitem_image_mode(self):
        from visual_search.evaluation.val_dataset import ValDataset
        ds = ValDataset()
        # Найдём первый image-запрос
        q = next(ds.iter_mode("image"))
        assert q.mode == "image"
        assert isinstance(q.target_image_ids, set)
        assert len(q.target_image_ids) > 0
        assert q.image_path is not None
        assert q.txt_query is None

    def test_getitem_txt_mode(self):
        from visual_search.evaluation.val_dataset import ValDataset
        ds = ValDataset()
        q = next(ds.iter_mode("txt"))
        assert q.mode == "txt"
        assert q.txt_query is not None
        assert len(q.target_image_ids) > 0

    def test_getitem_multimodal_mode(self):
        from visual_search.evaluation.val_dataset import ValDataset
        ds = ValDataset()
        q = next(ds.iter_mode("multimodal"))
        assert q.mode == "multimodal"
        assert q.image_path is not None
        assert q.txt_query is not None

    def test_evaluate_with_mock_search(self):
        """Smoke-тест: прогон evaluate_with_search_fn с заглушкой поиска."""
        from visual_search.evaluation.val_dataset import ValDataset
        from visual_search.evaluation.evaluate import evaluate_with_search_fn

        # Поиск всегда возвращает пустой список → все метрики 0
        results = evaluate_with_search_fn(
            search_fn=lambda q: [],
            k_values=[10],
            verbose=False,
        )
        assert "all" in results
        assert results["all"].recall_at_k[10] == pytest.approx(0.0)
        assert results["all"].mrr_score == pytest.approx(0.0)

    def test_evaluate_perfect_search(self):
        """Если поиск возвращает все таргеты первыми — mrr=1, recall@1000≈1."""
        from visual_search.evaluation.val_dataset import ValDataset
        from visual_search.evaluation.evaluate import evaluate_with_search_fn

        def perfect_search(q):
            # Возвращаем все таргеты на первых позициях
            return list(q.target_image_ids) + list(range(999_000, 999_000 + 50))

        results = evaluate_with_search_fn(
            search_fn=perfect_search,
            # k=1000 — гарантированно вместит все таргеты (max ~20 в датасете)
            k_values=[1000],
            verbose=False,
        )
        assert results["all"].recall_at_k[1000] == pytest.approx(1.0)
        assert results["all"].mrr_score == pytest.approx(1.0)
