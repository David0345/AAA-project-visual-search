"""Validation dataset loader for visual search evaluation."""

import os
from dataclasses import dataclass

import pandas as pd


VAL_DATASET_PATH = "../../val_dataset.csv"
IMAGES_BASE = "dataset_1M"


@dataclass
class ValQuery:
    query_id: int
    mode: str  # "image" | "txt" | "multimodal"
    item_id: int
    image_path: str | None  # absolute path to query image
    txt_query: str | None  # text query (txt & multimodal modes)
    target_image_ids: list[int]  # ground-truth image IDs
    metadata: dict  # param2, cvet, brand, etc.


class ValDataset:
    def __init__(self, csv_path: str = VAL_DATASET_PATH, images_base: str = IMAGES_BASE):
        self.df = pd.read_csv(csv_path)
        self.images_base = images_base

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ValQuery:
        row = self.df.iloc[idx]
        target_ids = (
            [int(x) for x in str(row["target_images_id"]).split(";") if x]
            if pd.notna(row["target_images_id"])
            else []
        )
        image_path = (
            os.path.join(self.images_base, row["image_path"])
            if pd.notna(row["image_path"])
            else None
        )
        return ValQuery(
            query_id=int(row["query_id"]),
            mode=row["mode"],
            item_id=int(row["item_id"]),
            image_path=image_path,
            txt_query=row["txt_query"] if pd.notna(row["txt_query"]) else None,
            target_image_ids=target_ids,
            metadata={
                "param2": row["param2"],
                "category_name": row["category_name"],
                "sostoyanie": row["sostoyanie"],
                "cvet": row["cvet"],
                "brand": row["brand"],
            },
        )

    def iter_mode(self, mode: str):
        """Iterate over queries of a specific mode ('image', 'txt', 'multimodal')."""
        for idx in self.df.index[self.df["mode"] == mode]:
            yield self[idx]

    def get_by_mode(self, mode: str) -> list[ValQuery]:
        """Return all queries of a specific mode."""
        return list(self.iter_mode(mode))


def recall_at_k(ranked_ids: list[int], target_ids: list[int], k: int) -> float:
    """Recall@K: fraction of targets found in top-K results."""
    if not target_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & set(target_ids)) / len(target_ids)


def mrr(ranked_ids: list[int], target_ids: list[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    target_set = set(target_ids)
    for i, rid in enumerate(ranked_ids):
        if rid in target_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    search_fn,
    dataset: ValDataset,
    mode: str | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """
    Evaluate a search function on the validation dataset.

    Args:
        search_fn: callable(ValQuery) -> list[int]
            Takes a query and returns ranked list of image IDs.
        dataset: ValDataset instance.
        mode: filter by mode ('image', 'txt', 'multimodal') or None for all.
        k_values: list of K for Recall@K (default: [1, 5, 10]).

    Returns:
        dict with metrics: recall@k, mrr, count, per-category breakdown.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    queries = dataset.get_by_mode(mode) if mode else [dataset[i] for i in range(len(dataset))]

    recalls = {k: [] for k in k_values}
    mrrs = []
    per_category = {}

    for q in queries:
        ranked = search_fn(q)
        targets = q.target_image_ids

        m = mrr(ranked, targets)
        mrrs.append(m)
        for k in k_values:
            recalls[k].append(recall_at_k(ranked, targets, k))

        cat = q.metadata["param2"]
        if cat not in per_category:
            per_category[cat] = {"mrr": [], **{f"r@{k}": [] for k in k_values}}
        per_category[cat]["mrr"].append(m)
        for k in k_values:
            per_category[cat][f"r@{k}"].append(recall_at_k(ranked, targets, k))

    results = {
        "mode": mode or "all",
        "count": len(queries),
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
    }
    for k in k_values:
        results[f"recall@{k}"] = sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0

    results["per_category"] = {
        cat: {
            metric: sum(vals) / len(vals) if vals else 0
            for metric, vals in metrics.items()
        }
        for cat, metrics in per_category.items()
    }
    return results


if __name__ == "__main__":
    ds = ValDataset()
    print(f"Total queries: {len(ds)}")
    for mode in ["image", "txt", "multimodal"]:
        queries = ds.get_by_mode(mode)
        print(f"  {mode}: {len(queries)}")

    q = ds[0]
    print(f"\nExample query: mode={q.mode}, targets={len(q.target_image_ids)}")
    print(f"  image_path={q.image_path}")
    print(f"  txt_query={q.txt_query}")
    print(f"  metadata={q.metadata}")
