"""Query pipeline: encode -> ANN-поиск -> hydrate метаданных (URL, заголовок).

Тот же путь, что прогоняется в оценке. Опциональный re-ranker — здесь, после ANN.

TODO(деплой): search(query) -> top-N результатов с метаданными.
"""

from __future__ import annotations
