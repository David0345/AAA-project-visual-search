"""Обёртка над ANN-бэкендом (faiss) — единый интерфейс поиска.

Бэкенд и стратегия квантования — точка тюнинга под < 1 сек / 1 rps; интерфейс
search(query_vec, k) -> [(item_id, score), ...] от этого не зависит. Какой
бэкенд/квантование выбрать — решает scripts/benchmark_index.py.

Эмбеддинги L2-нормированы, поэтому метрика — inner product (= косинус).

Поддерживаемые бэкенды (preset -> faiss-индекс):
  * "flat"  — IndexFlatIP: точный, эталон качества, ~4*N*D байт в RAM;
  * "ivf"   — IndexIVFFlat: инвертированные списки, быстрый, без сжатия;
  * "ivfpq" — IndexIVFPQ: IVF + product quantization, сильное сжатие RAM;
  * "hnsw"  — IndexHNSWFlat: граф, очень быстрый поиск, больше RAM.

Сохранение на диск — директория:
  <dir>/index.faiss   бинарный индекс
  <dir>/ids.npy       item_id, параллельный строкам индекса
  <dir>/meta.json     embed_dim, backend, params, num_vectors
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visual_search.common.logging import get_logger

log = get_logger(__name__)

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _require_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "faiss не установлен. Поставьте: uv add faiss-cpu (или faiss-gpu)."
        )


@dataclass
class IndexSpec:
    """Конфигурация ANN-индекса. nlist/m рассчитываются от N, если None."""

    backend: str = "flat"
    nlist: int | None = None       # IVF: число ячеек Вороного
    nprobe: int = 32               # IVF: сколько ячеек смотреть при поиске
    m: int | None = None           # PQ: число суб-квантователей (делит embed_dim)
    nbits: int = 8                 # PQ: бит на суб-вектор
    hnsw_m: int = 32               # HNSW: связей на узел
    ef_search: int = 64            # HNSW: ширина поиска
    ef_construction: int = 200     # HNSW: ширина построения
    extra: dict[str, Any] = field(default_factory=dict)


def _auto_nlist(n: int) -> int:
    """~4*sqrt(N), но не больше N//39 (faiss хочет >=39 точек обучения на ячейку)."""
    return max(1, min(round(4 * math.sqrt(max(n, 1))), max(1, n // 39)))


def _largest_divisor_leq(dim: int, cap: int) -> int:
    """Наибольший делитель dim, не превышающий cap (для PQ: m должно делить dim)."""
    for cand in range(min(cap, dim), 0, -1):
        if dim % cand == 0:
            return cand
    return 1


def build_faiss_index(spec: IndexSpec, embed_dim: int, n: int):
    """Собрать пустой faiss-индекс по спецификации (metric = inner product)."""
    _require_faiss()
    metric = faiss.METRIC_INNER_PRODUCT
    backend = spec.backend.lower()

    if backend == "flat":
        return faiss.IndexFlatIP(embed_dim)

    if backend in ("ivf", "ivfpq"):
        nlist = spec.nlist or _auto_nlist(n)
        quantizer = faiss.IndexFlatIP(embed_dim)
        if backend == "ivf":
            index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, metric)
        else:
            m = spec.m or _largest_divisor_leq(embed_dim, 64)
            if embed_dim % m != 0:
                m = _largest_divisor_leq(embed_dim, m)
            index = faiss.IndexIVFPQ(quantizer, embed_dim, nlist, m, spec.nbits, metric)
        index.nprobe = spec.nprobe
        return index

    if backend == "hnsw":
        index = faiss.IndexHNSWFlat(embed_dim, spec.hnsw_m, metric)
        index.hnsw.efSearch = spec.ef_search
        index.hnsw.efConstruction = spec.ef_construction
        return index

    raise ValueError(f"Неизвестный backend {spec.backend!r}")


@dataclass
class ANNIndex:
    """ANN-индекс на уровне товара: строки -> item_id."""

    embed_dim: int
    spec: IndexSpec = field(default_factory=IndexSpec)
    _index: Any = field(default=None, repr=False)
    _ids: np.ndarray | None = field(default=None, repr=False)

    # -- построение -------------------------------------------------------
    def build(self, vectors: np.ndarray, ids: np.ndarray) -> "ANNIndex":
        """vectors: (N, embed_dim) float32 L2-norm; ids: (N,) item_id."""
        _require_faiss()
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.embed_dim:
            raise ValueError(f"embed_dim {vectors.shape[1]} != {self.embed_dim}")
        if len(vectors) != len(ids):
            raise ValueError("len(vectors) != len(ids)")

        index = build_faiss_index(self.spec, self.embed_dim, len(vectors))
        if not index.is_trained:
            log.info("Тренировка индекса (%s) на %d векторах...", self.spec.backend, len(vectors))
            index.train(vectors)
        index.add(vectors)
        self._index = index
        self._ids = np.ascontiguousarray(ids, dtype=np.int64)
        log.info("Индекс построен: %d векторов, backend=%s", index.ntotal, self.spec.backend)
        return self

    # -- поиск ------------------------------------------------------------
    def batch_search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """queries: (Q, embed_dim) -> (ids (Q,k) item_id, scores (Q,k))."""
        if self._index is None or self._ids is None:
            raise RuntimeError("Индекс не построен/не загружен")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        scores, positions = self._index.search(queries, k)
        ids = np.where(positions >= 0, self._ids[positions.clip(min=0)], -1)
        return ids, scores

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """query: (embed_dim,) -> [(item_id, score), ...] по убыванию score."""
        ids, scores = self.batch_search(query.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    # -- персистентность --------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Сохранить в директорию: index.faiss + ids.npy + meta.json."""
        _require_faiss()
        if self._index is None or self._ids is None:
            raise RuntimeError("Нечего сохранять: индекс не построен")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out / "index.faiss"))
        np.save(out / "ids.npy", self._ids)
        meta = {
            "embed_dim": self.embed_dim,
            "backend": self.spec.backend,
            "num_vectors": int(self._index.ntotal),
            "nprobe": self.spec.nprobe,
            "nlist": self.spec.nlist,
            "m": self.spec.m,
            "nbits": self.spec.nbits,
            "hnsw_m": self.spec.hnsw_m,
            "ef_search": self.spec.ef_search,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("Индекс сохранён: %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ANNIndex":
        """Загрузить индекс из директории (для evaluation и serving)."""
        _require_faiss()
        src = Path(path)
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        spec = IndexSpec(
            backend=meta["backend"],
            nlist=meta.get("nlist"),
            nprobe=meta.get("nprobe", 32),
            m=meta.get("m"),
            nbits=meta.get("nbits", 8),
            hnsw_m=meta.get("hnsw_m", 32),
            ef_search=meta.get("ef_search", 64),
        )
        obj = cls(embed_dim=meta["embed_dim"], spec=spec)
        obj._index = faiss.read_index(str(src / "index.faiss"))
        obj._ids = np.load(src / "ids.npy")
        return obj

    # -- утилиты для бенчмарка -------------------------------------------
    def size_bytes(self) -> int:
        """Размер сериализованного индекса в байтах (оценка RAM/диска)."""
        _require_faiss()
        if self._index is None:
            return 0
        return int(faiss.serialize_index(self._index).nbytes)

    @property
    def ntotal(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)
