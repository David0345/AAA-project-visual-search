"""FastAPI-приложение сервиса поиска (каркас).

Поднимается тонкой обёрткой scripts/serve.py. Модель и индекс грузятся один раз
при старте; пути приходят из конфига/переменных окружения (под облако).

TODO(деплой): эндпойнты /search/image, /search/text, /search/multimodal, /health.
"""

from __future__ import annotations
