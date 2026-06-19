"""Загрузка валидационного датасета.

CSV-схема:
    query_id, mode, item_id, image_id, image_path, txt_query,
    target_images_id, param2, category_name, sostoyanie, cvet, brand

Режимы:
    image       — запрос по картинке; image_path заполнен, txt_query пуст
    txt         — текстовый запрос;   txt_query заполнен, image_path пуст
    multimodal  — оба поля заполнены

target_images_id хранится как строка вида ``"{id1, id2, id3}"`` (Python set repr).

Пример использования::

    from visual_search.evaluation.val_dataset import ValDataset
    ds = ValDataset("path/to/val_dataset.csv", images_base="data/raw/dataset_1M")
    for q in ds.iter_mode("txt"):
        print(q.txt_query, q.target_image_ids)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import pandas as pd


# Путь по умолчанию — рядом с этим файлом лежит папка val_dataset/
_DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "val_dataset", "val_dataset.csv")


@dataclass
class ValQuery:
    """Один запрос из валидационного датасета."""

    query_id: int
    mode: str                    # "image" | "txt" | "multimodal"
    item_id: int
    image_id: int | None         # image_id запроса (не таргета)
    image_path: str | None       # абсолютный путь к картинке запроса
    txt_query: str | None        # текстовый запрос (txt & multimodal)
    target_image_ids: set[int]   # ground-truth image_id
    metadata: dict               # param2, category_name, cvet, brand, sostoyanie


def _parse_target_ids(raw: str) -> set[int]:
    """Разобрать ``"{id1, id2, ...}"`` → ``{id1, id2, ...}``.

    Поддерживает три формата, которые встречались в датасете:
      * ``{1, 2, 3}``  — Python set repr (основной формат CSV)
      * ``1;2;3``       — разделитель «;» (устаревший)
      * ``1,2,3``       — просто запятая
    """
    raw = raw.strip()
    if not raw:
        return set()

    # Основной формат: {id1, id2, ...}
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1]
        return {int(x.strip()) for x in inner.split(",") if x.strip()}

    # Пробуем ast.literal_eval (set/frozenset/list/tuple)
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (set, frozenset, list, tuple)):
            return {int(x) for x in val}
    except (ValueError, SyntaxError):
        pass

    # Fallback: разделитель ; или ,
    sep = ";" if ";" in raw else ","
    return {int(x.strip()) for x in raw.split(sep) if x.strip()}


class ValDataset:
    """Загружает val_dataset.csv и отдаёт ValQuery по индексу или режиму."""

    def __init__(
        self,
        csv_path: str = _DEFAULT_CSV,
        images_base: str = "",
    ) -> None:
        """
        Args:
            csv_path:    путь к val_dataset.csv.
            images_base: префикс, который приклеивается к image_path из CSV.
                         Например: ``"data/raw/dataset_1M"``.
                         Если пустая строка — пути остаются как в CSV.
        """
        self.df = pd.read_csv(csv_path)
        self.images_base = images_base

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ValQuery:
        row = self.df.iloc[idx]

        target_ids: set[int] = (
            _parse_target_ids(str(row["target_images_id"]))
            if pd.notna(row.get("target_images_id"))
            else set()
        )

        raw_path: str | None = str(row["image_path"]) if pd.notna(row.get("image_path")) else None
        image_path: str | None = (
            os.path.join(self.images_base, raw_path)
            if (raw_path and self.images_base)
            else raw_path
        )

        return ValQuery(
            query_id=int(row["query_id"]),
            mode=str(row["mode"]),
            item_id=int(row["item_id"]),
            image_id=int(row["image_id"]) if pd.notna(row.get("image_id")) else None,
            image_path=image_path,
            txt_query=str(row["txt_query"]) if pd.notna(row.get("txt_query")) else None,
            target_image_ids=target_ids,
            metadata={
                "param2": row.get("param2"),
                "category_name": row.get("category_name"),
                "sostoyanie": row.get("sostoyanie"),
                "cvet": row.get("cvet"),
                "brand": row.get("brand"),
            },
        )

    def iter_mode(self, mode: str):
        """Итерировать запросы конкретного режима ('image', 'txt', 'multimodal')."""
        for pos in range(len(self.df)):
            if self.df.iloc[pos]["mode"] == mode:
                yield self[pos]

    def get_by_mode(self, mode: str) -> list[ValQuery]:
        """Вернуть все запросы конкретного режима списком."""
        return list(self.iter_mode(mode))

    def stats(self) -> dict[str, int]:
        """Краткая статистика датасета."""
        return {
            "total": len(self),
            **{m: int((self.df["mode"] == m).sum()) for m in ["image", "txt", "multimodal"]},
        }


if __name__ == "__main__":
    ds = ValDataset()
    st = ds.stats()
    print(
        f"ValDataset: total={st['total']}  "
        f"image={st['image']}  txt={st['txt']}  multimodal={st['multimodal']}"
    )
    q = ds[0]
    print(f"\nПример: mode={q.mode!r}, query_id={q.query_id}")
    print(f"  image_path : {q.image_path}")
    print(f"  txt_query  : {q.txt_query}")
    print(f"  targets    : {q.target_image_ids}")
    print(f"  metadata   : {q.metadata}")
