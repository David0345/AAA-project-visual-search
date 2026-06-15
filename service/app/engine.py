"""Поисковый движок: загрузка модели-энкодера и FAISS-индекса каталога,
кодирование запроса (image / txt / multimodal) и поиск ближайших товаров.

Каждый метод делает одно действие → легко тестировать."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image

# пакет visual_search лежит в src/ репозитория
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from visual_search.models.registry import build_model
from visual_search.models import encoders  # noqa: F401 — регистрация моделей

from .config import Settings
from .schemas import SearchHit, SearchMode


def load_encoder(model_name: str, checkpoint_path: str | None, device: str):
    """Собрать энкодер из registry и (опц.) загрузить веса дообучения."""
    model = build_model({"name": model_name}).to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    return model


def decode_image(image_b64: str) -> Image.Image:
    """base64 → PIL.Image (RGB)."""
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


class SearchEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(settings.device)
        self.model = load_encoder(settings.model_name, settings.checkpoint_path, settings.device)
        self.index = faiss.read_index(settings.index_path)
        self.meta = pd.read_parquet(settings.meta_path)   # строка i ↔ вектор i

    # -- кодирование --------------------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.model.tokenize(text).to(self.device)
            return self.model.encode_text(tokens).squeeze(0).cpu().numpy()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            t = self.model.preprocess_image(image).to(self.device)
            return self.model.encode_image(t).squeeze(0).cpu().numpy()

    def encode_query(self, mode: SearchMode, text: str | None, image: Image.Image | None) -> np.ndarray:
        """Собрать вектор запроса по режиму; multimodal — взвешенная склейка."""
        vecs, weights = [], []
        w_img = self.settings.mm_image_weight
        if image is not None and mode in (SearchMode.image, SearchMode.multimodal):
            vecs.append(self.encode_image(image))
            weights.append(w_img if mode == SearchMode.multimodal else 1.0)
        if text and mode in (SearchMode.txt, SearchMode.multimodal):
            vecs.append(self.encode_text(text))
            weights.append((1.0 - w_img) if mode == SearchMode.multimodal else 1.0)
        if not vecs:
            raise ValueError(f"Недостаточно данных для режима {mode.value}")
        return l2_normalize(np.average(vecs, axis=0, weights=weights).astype(np.float32))

    # -- поиск --------------------------------------------------------------
    def search(self, vec: np.ndarray, top_k: int) -> list[SearchHit]:
        scores, idxs = self.index.search(vec[None, :], top_k)
        hits = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            row = self.meta.iloc[int(i)]
            hits.append(SearchHit(item_id=int(row["item_id"]),
                                  image_path=row.get("image_path"),
                                  score=float(score)))
        return hits
