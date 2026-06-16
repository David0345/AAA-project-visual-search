#!/usr/bin/env python3
"""CLI: запуск веб-сервиса поиска.

Тонкая обёртка над uvicorn + visual_search.serving.app:app.
Параметры модели/индекса/бакета берутся из переменных окружения
(см. visual_search.serving.search.EngineConfig).

Запуск:
    python scripts/serve.py                      # 0.0.0.0:8000
    HOST=127.0.0.1 PORT=8080 python scripts/serve.py
    # или напрямую:
    uvicorn visual_search.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "visual_search.serving.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
    )


if __name__ == "__main__":
    main()
