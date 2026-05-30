"""Smoke-тест: пакет импортируется, контракты на месте.

Расширяется по подпакету (по одному test-файлу на модуль). Реальные тесты
добавляют владельцы модулей.
"""

from __future__ import annotations


def test_package_imports():
    import visual_search
    from visual_search.models import Encoder, build_model  # noqa: F401

    assert visual_search.__version__
