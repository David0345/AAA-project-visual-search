"""Логирование метрик прогона в CSV (experiments/<run_id>/metrics.csv).

Бэкенд — csv; итоговая сводка — summary.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from visual_search.common.logging import get_logger

log = get_logger(__name__)


class MetricsTracker:

    def __init__(self, log_dir: str | Path) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / "metrics.csv"
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._t0 = time.monotonic()

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {
            "step": step,
            "wall_sec": round(time.monotonic() - self._t0, 1),
            **metrics,
        }
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=list(row.keys()),
                restval="",
                extrasaction="ignore",
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def log_summary(self, summary: dict[str, Any]) -> None:
        path = self._dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info("Summary saved: %s", path)

    def close(self) -> None:
        self._file.close()
