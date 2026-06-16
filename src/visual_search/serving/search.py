"""Query pipeline сервиса: encode -> ANN-поиск -> hydrate метаданных + image_url.

SearchEngine грузит модель/индекс/метаданные один раз (при старте сервиса) и на
каждый запрос: кодирует текст/картинку/мультимодальный запрос, ищет top-K в
ANN-индексе (на уровне товара, ключ = item_id), подтягивает метаданные и
формирует image_url (presigned для приватного бакета или публичный base+path).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from visual_search.index.ann import ANNIndex
from visual_search.serving.presign import presign_get

log = logging.getLogger(__name__)


def _feat(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"Не нашёл эмбеддинг в выводе {type(out).__name__}")


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EngineConfig:
    model_dir: str = "google/siglip2-base-patch16-224"
    index_dir: str = "data/processed/serving_index"
    device: str = "auto"
    max_text_len: int = 64
    mm_alpha: float = 0.5
    image_url_mode: str = "presign"          # presign | public
    s3_bucket: str = ""
    s3_endpoint: str = "storage.yandexcloud.net"
    s3_region: str = "ru-central1"
    url_expires: int = 3600
    public_base_url: str = ""

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            model_dir=os.getenv("MODEL_DIR", cls.model_dir),
            index_dir=os.getenv("INDEX_DIR", cls.index_dir),
            device=os.getenv("DEVICE", cls.device),
            max_text_len=int(os.getenv("MAX_TEXT_LEN", cls.max_text_len)),
            mm_alpha=float(os.getenv("MM_ALPHA", cls.mm_alpha)),
            image_url_mode=os.getenv("IMAGE_URL_MODE", cls.image_url_mode),
            s3_bucket=os.getenv("S3_BUCKET", ""),
            s3_endpoint=os.getenv("S3_ENDPOINT", cls.s3_endpoint),
            s3_region=os.getenv("S3_REGION", cls.s3_region),
            url_expires=int(os.getenv("IMAGE_URL_EXPIRES", cls.url_expires)),
            public_base_url=os.getenv("PUBLIC_BASE_URL", ""),
        )


class SearchEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        log.info("SearchEngine device=%s", self.device)

        # faiss + torch в одном процессе → один поток faiss во избежание segfault
        try:
            import faiss
            faiss.omp_set_num_threads(1)
        except Exception:  # noqa: BLE001
            pass

        from transformers import AutoModel, AutoProcessor
        log.info("Загружаю модель %s ...", cfg.model_dir)
        self.processor = AutoProcessor.from_pretrained(cfg.model_dir)
        self.model = AutoModel.from_pretrained(cfg.model_dir).to(self.device).eval()

        log.info("Загружаю индекс %s ...", cfg.index_dir)
        self.index = ANNIndex.load(cfg.index_dir)
        meta_path = Path(cfg.index_dir) / "metadata.parquet"
        md = pd.read_parquet(meta_path)
        md["item_id"] = md["item_id"].astype("int64")
        self.meta = md.set_index("item_id").to_dict("index")
        log.info("Готово: %d векторов, %d записей метаданных", self.index.ntotal, len(self.meta))

        # ключи S3 для presign (если режим presign)
        self._ak = os.getenv("AWS_ACCESS_KEY_ID", "")
        self._sk = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        if cfg.image_url_mode == "presign" and not (self._ak and self._sk and cfg.s3_bucket):
            log.warning("image_url_mode=presign, но нет S3-ключей/бакета — image_url будут как public/относительные")

    # ---- кодирование запросов ----
    @torch.no_grad()
    def _embed_text(self, text: str) -> np.ndarray:
        enc = self.processor(text=[text], padding="max_length", max_length=self.cfg.max_text_len,
                             truncation=True, return_tensors="pt").to(self.device)
        v = F.normalize(_feat(self.model.get_text_features(**enc)).float(), dim=-1)
        return v.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _embed_image(self, img: Image.Image) -> np.ndarray:
        enc = self.processor(images=[img.convert("RGB")], return_tensors="pt").to(self.device)
        v = F.normalize(_feat(self.model.get_image_features(**enc)).float(), dim=-1)
        return v.squeeze(0).cpu().numpy().astype(np.float32)

    def _query_vector(self, text: str | None, image: Image.Image | None) -> tuple[str, np.ndarray]:
        vecs, weights, mode = [], [], None
        if image is not None and text:
            mode = "multimodal"
            vecs = [self._embed_image(image), self._embed_text(text)]
            weights = [1.0 - self.cfg.mm_alpha, self.cfg.mm_alpha]
        elif image is not None:
            mode = "image"
            vecs, weights = [self._embed_image(image)], [1.0]
        elif text:
            mode = "text"
            vecs, weights = [self._embed_text(text)], [1.0]
        else:
            raise ValueError("Пустой запрос: нужен text и/или image")
        vec = np.average(vecs, axis=0, weights=weights).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 1e-8:
            vec = vec / n
        return mode, vec

    # ---- image_url ----
    def _image_url(self, image_path: str) -> str:
        c = self.cfg
        if c.image_url_mode == "presign" and self._ak and self._sk and c.s3_bucket:
            return presign_get(c.s3_bucket, image_path, self._ak, self._sk,
                               host=c.s3_endpoint, region=c.s3_region, expires=c.url_expires)
        if c.public_base_url:
            return c.public_base_url.rstrip("/") + "/" + image_path.lstrip("/")
        return image_path  # fallback (например, для локального теста)

    # ---- поиск ----
    def search(self, text: str | None, image: Image.Image | None, top_k: int = 10) -> tuple[str, list[dict]]:
        mode, vec = self._query_vector(text, image)
        ids, scores = self.index.batch_search(vec[None, :], k=top_k)
        results = []
        for iid, sc in zip(ids[0], scores[0]):
            iid = int(iid)
            if iid == -1:
                continue
            md = self.meta.get(iid, {})
            results.append({
                "item_id": iid,
                "score": float(sc),
                "image_url": self._image_url(str(md.get("image_path", ""))),
                "title": str(md.get("product_text", "")),
                "param2": (str(md["param2"]) if md.get("param2") is not None else None),
                "brand": (str(md["brand"]) if md.get("brand") is not None else None),
            })
        return mode, results
