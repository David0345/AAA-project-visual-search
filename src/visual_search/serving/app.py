"""FastAPI-приложение сервиса визуального поиска.

Эндпойнты:
    POST /api/search   — multipart/form-data: text?, image?, top_k?(=10), mode?(опц.)
                         -> {status, query_mode, results:[{item_id,score,image_url,title,param2,brand}]}
    GET  /health       — статус + размер каталога
    GET  /             — отдаёт фронтенд (serving/index.html)

Модель/индекс/метаданные грузятся один раз при старте (пути из env, см.
visual_search.serving.search.EngineConfig). Запуск: scripts/serve.py или
`uvicorn visual_search.serving.app:app`.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from visual_search.serving.schemas import SearchResponse, SearchResult
from visual_search.serving.search import EngineConfig, SearchEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_INDEX_HTML = Path(__file__).parent / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = EngineConfig.from_env()
    app.state.engine = SearchEngine(cfg)   # грузим модель/индекс один раз
    yield


app = FastAPI(title="AAA Visual Search", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    eng = getattr(app.state, "engine", None)
    return JSONResponse({
        "status": "ok" if eng else "loading",
        "items": eng.index.ntotal if eng else 0,
        "model": eng.cfg.model_dir if eng else None,
    })


@app.post("/api/search", response_model=SearchResponse)
async def search(
    text: str | None = Form(None),
    image: UploadFile | None = File(None),
    top_k: int = Form(10),
    mode: str | None = Form(None),          # фронт его шлёт; режим всё равно выводим из полей
) -> SearchResponse:
    eng: SearchEngine = app.state.engine
    text = (text or "").strip() or None
    top_k = max(1, min(int(top_k or 10), 100))

    pil = None
    if image is not None:
        data = await image.read()
        if data:
            try:
                pil = Image.open(io.BytesIO(data))
                pil.load()
            except Exception:  # noqa: BLE001
                raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    if pil is None and not text:
        raise HTTPException(status_code=400, detail="Нужен text и/или image")

    try:
        # инференс блокирующий (torch/faiss) — уводим из event loop в threadpool
        query_mode, results = await run_in_threadpool(eng.search, text, pil, top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SearchResponse(
        status="success",
        query_mode=query_mode,
        results=[SearchResult(**r) for r in results],
    )


@app.get("/")
def index() -> FileResponse:
    if not _INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html не найден")
    return FileResponse(_INDEX_HTML)
