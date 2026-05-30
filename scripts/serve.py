"""CLI: запуск веб-сервиса поиска (этап деплоя).

Тонкая обёртка — приложение в visual_search.serving.app.
Запуск: python scripts/serve.py  (или uvicorn visual_search.serving.app:app)
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("TODO(деплой): поднять uvicorn с visual_search.serving.app")


if __name__ == "__main__":
    main()
