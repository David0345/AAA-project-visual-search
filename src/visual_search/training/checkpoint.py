"""Сохранение/загрузка чекпойнтов в experiments/<run_id>/checkpoints/.

Чекпойнт несёт веса + имя модели/конфиг, чтобы build_model мог восстановить
точно ту же архитектуру при оценке и в сервисе.

TODO(Обучение): save_checkpoint / load_checkpoint.
"""

from __future__ import annotations
