"""Логирование метрик прогона: csv / TensorBoard / Weights & Biases.

Бэкенд выбирается из конфига; по умолчанию — csv в experiments/<run_id>/.

TODO(Обучение): единый интерфейс log_metrics(step, {...}).
"""

from __future__ import annotations
