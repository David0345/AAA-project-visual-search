"""FastAPI-сервис визуального поиска по каталогу Avito (image / txt / multimodal)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import get_settings
from .engine import SearchEngine, decode_image
from .schemas import SearchMode, SearchRequest, SearchResponse

_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = SearchEngine(get_settings())   # грузим модель+индекс один раз на старте
    yield


app = FastAPI(title="Avito Visual Search", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if _engine is None:
        raise HTTPException(503, "engine not ready")
    image = decode_image(req.image_b64) if req.image_b64 else None
    try:
        vec = _engine.encode_query(req.mode, req.text, image)
    except ValueError as e:
        raise HTTPException(422, str(e))
    top_k = req.top_k or get_settings().top_k
    return SearchResponse(mode=req.mode, hits=_engine.search(vec, top_k))
