"""Логирование метрик прогона.

Бэкенды:
  * CSV + summary.json — всегда включён (experiments/<run_id>/metrics.csv)
  * WandB              — включается через конфиг: wandb.enabled=true
                         или env-переменную WANDB_PROJECT

Конфиг (секция wandb в config.yaml):
    wandb:
      enabled: true          # или false
      project: avito-visual-search
      entity: null           # ваша org/username в wandb
      tags: []
      notes: ""
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from visual_search.common.logging import get_logger

log = get_logger(__name__)


class MetricsTracker:
    """Пишет метрики в CSV и опционально в WandB.

    Args:
        log_dir:      директория для CSV/summary.json (обычно Hydra output_dir).
        wandb_config: DictConfig / dict с ключами enabled, project, entity,
                      tags, notes, run_name. Если None — только CSV.
        run_config:   весь Hydra-конфиг эксперимента для логирования в wandb.
    """

    def __init__(
        self,
        log_dir: str | Path,
        wandb_config: Any | None = None,
        run_config: Any | None = None,
    ) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / "metrics.csv"
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._t0 = time.monotonic()
        self._wandb_run = None

        self._init_wandb(wandb_config, run_config)

    # ------------------------------------------------------------------
    # WandB initialization
    # ------------------------------------------------------------------

    def _init_wandb(self, wandb_config: Any, run_config: Any) -> None:
        """Инициализировать wandb, если включён."""
        # Проверяем: явный конфиг или env-переменная
        enabled = False
        project = os.environ.get("WANDB_PROJECT")

        if wandb_config is not None:
            try:
                enabled = bool(wandb_config.get("enabled", False))
                project = wandb_config.get("project", project)
            except (AttributeError, TypeError):
                pass

        if not enabled and not project:
            return  # WandB выключен

        try:
            import wandb
        except ImportError:
            log.warning("wandb не установлен. Логирование только в CSV. Поставьте: pip install wandb")
            return

        try:
            cfg_dict = None
            if run_config is not None:
                try:
                    from omegaconf import OmegaConf
                    cfg_dict = OmegaConf.to_container(run_config, resolve=True)
                except Exception:
                    cfg_dict = dict(run_config) if hasattr(run_config, "__iter__") else None

            run_name = None
            entity = None
            tags = []
            notes = ""
            if wandb_config is not None:
                try:
                    run_name = wandb_config.get("run_name") or str(self._dir.parent.name)
                    entity = wandb_config.get("entity")
                    tags = list(wandb_config.get("tags", []))
                    notes = str(wandb_config.get("notes", ""))
                except (AttributeError, TypeError):
                    pass

            self._wandb_run = wandb.init(
                project=project or "avito-visual-search",
                name=run_name,
                entity=entity or None,
                config=cfg_dict,
                tags=tags or None,
                notes=notes or None,
                dir=str(self._dir),
                resume="allow",
            )
            log.info("WandB run: %s", self._wandb_run.url)
        except Exception as exc:
            log.warning("Не удалось инициализировать WandB: %s. Логирование только в CSV.", exc)
            self._wandb_run = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Записать метрики шага в CSV и WandB."""
        row = {
            "step": step,
            "wall_sec": round(time.monotonic() - self._t0, 1),
            **metrics,
        }
        # CSV
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

        # WandB
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception as exc:
                log.debug("wandb.log error: %s", exc)

    def log_eval(self, step: int, eval_results: dict[str, Any]) -> None:
        """Записать результаты оценки (после epoch) с префиксом eval/.

        eval_results: {mode: ModeMetrics} или плоский dict.
        """
        flat: dict[str, Any] = {}
        for key, val in eval_results.items():
            if hasattr(val, "as_flat_dict"):
                for k, v in val.as_flat_dict().items():
                    if k not in ("mode", "count"):
                        flat[f"eval/{key}/{k}"] = v
            else:
                flat[f"eval/{key}"] = val

        self.log(step, flat)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Записать итоговую сводку в summary.json (и wandb.summary)."""
        path = self._dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info("Summary saved: %s", path)

        if self._wandb_run is not None:
            try:
                self._wandb_run.summary.update(summary)
            except Exception:
                pass

    def close(self) -> None:
        """Закрыть файл и завершить wandb run."""
        self._file.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
