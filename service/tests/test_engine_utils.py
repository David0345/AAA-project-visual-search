"""Тесты чистых функций движка (без загрузки модели/индекса)."""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.engine import decode_image, l2_normalize


def test_l2_normalize_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_l2_normalize_zero_vector_safe():
    v = np.zeros(4, dtype=np.float32)
    assert np.allclose(l2_normalize(v), v)   # без деления на ноль


def test_decode_image_roundtrip():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img = decode_image(b64)
    assert img.mode == "RGB" and img.size == (8, 8)
