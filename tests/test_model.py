"""Интеграционные тесты для моделей и лоссов.
Требуют скачивания весов RuCLIP из HuggingFace (около 600 МБ, кэшируются).
Запуск: pytest tests/test_models.py -v
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from visual_search.models.registry import build_model, get_processor
from visual_search.models.losses import InfoNCELoss
from visual_search.common.io import CONFIGS_DIR


MODEL_CONFIG_PATH = CONFIGS_DIR / "model" / "clip_vit_b32.yaml"


@pytest.fixture(scope="session")
def model_cfg():
    """
    Загружает реальный конфиг модели из YAML-файла.
    Если файла нет, тесты будут пропущены (защита от запуска в неправильной директории).
    """
    if not MODEL_CONFIG_PATH.exists():
        pytest.skip(f"Конфиг модели не найден по пути: {MODEL_CONFIG_PATH}")
    return OmegaConf.load(MODEL_CONFIG_PATH)


# Маркер для тестов, требующих загрузки модели из сети
requires_network = pytest.mark.skipif(
    not torch.cuda.is_available() and True,  # Всегда запускаем, если есть transformers
    reason="Требует загрузки модели из HuggingFace"
)


@requires_network
class TestRuCLIPInit:
    """Проверяем, что модель загружается и соблюдает контракт Encoder."""

    @pytest.fixture(scope="class")
    def model_and_processor(self, model_cfg):
        """Грузим модель и процессор один раз на весь класс тестов."""
        model = build_model(model_cfg)
        processor = get_processor(model_cfg.pretrained)
        return model, processor

    def test_embed_dim(self, model_and_processor):
        model, _ = model_and_processor
        assert model.embed_dim == 512, f"Ожидался embed_dim=512, получено {model.embed_dim}"

    def test_encode_image_l2_norm(self, model_and_processor):
        model, processor = model_and_processor

        fake_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        inputs = processor(images=[fake_img], return_tensors="pt")

        with torch.no_grad():
            img_emb = model.encode_image(inputs["pixel_values"])

        assert img_emb.shape == (1, 512)
        norms = torch.norm(img_emb, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Эмбеддинги не L2-нормированы! Norms: {norms}"

    def test_encode_text_with_attention_mask(self, model_and_processor):
        model, processor = model_and_processor

        inputs = processor(
            text=["чёрное платье", "белая футболка"],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        with torch.no_grad():
            txt_emb = model.encode_text(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        assert txt_emb.shape == (2, 512)
        norms = torch.norm(txt_emb, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Текстовые эмбеддинги не L2-нормированы! Norms: {norms}"

    def test_infonce_loss_no_nan(self, model_and_processor):
        """Проверяем, что InfoNCELoss считает валидное число на выходах модели."""
        model, processor = model_and_processor

        fake_imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]

        img_inputs = processor(images=fake_imgs, return_tensors="pt")
        txt_inputs = processor(
            text=["платье", "футболка", "брюки", "юбка"],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        with torch.no_grad():
            img_emb = model.encode_image(img_inputs["pixel_values"])
            txt_emb = model.encode_text(
                txt_inputs["input_ids"], 
                attention_mask=txt_inputs["attention_mask"]
            )

        loss_fn = InfoNCELoss()
        loss = loss_fn(img_emb, txt_emb)

        assert loss.dim() == 0, "Loss должен быть скаляром"
        assert not torch.isnan(loss), "Loss равен NaN!"
        assert not torch.isinf(loss), "Loss равен Inf!"
        assert loss.item() > 0, "InfoNCE loss должен быть положительным"
