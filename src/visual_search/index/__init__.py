"""index — эмбеддинги каталога и ANN-поиск (контракт §5.4).

ОБЩИЙ слой: используется и оценкой (evaluation/), и сервисом (serving/). Один и
тот же индекс гарантирует, что офлайн-метрики соответствуют онлайн-поведению.
Меняется по согласованию.

Публичный API:
    ANNIndex, IndexSpec   — обёртка над faiss (build/search/save/load);
    embed_catalog         — эмбеддинги каталога на уровне товара;
    build_index           — чекпойнт + каталог -> сохранённый индекс;
    run_benchmark         — подбор pooling/алгоритма/квантования по валидации.
"""

from visual_search.index.ann import ANNIndex, IndexSpec
from visual_search.index.build_index import build_index
from visual_search.index.embed import CatalogEmbeddings, embed_catalog, embed_images

__all__ = [
    "ANNIndex",
    "IndexSpec",
    "CatalogEmbeddings",
    "embed_catalog",
    "embed_images",
    "build_index",
]
