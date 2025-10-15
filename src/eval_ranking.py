"""Ranking evaluation for the ALS recommender."""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from . import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranking metrics on the test split.")
    parser.add_argument("--seed", type=int, default=utils.SEED, help="Random seed (unused but kept for symmetry).")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for precision/recall/NDCG.")
    return parser.parse_args()


def load_artifacts() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model_path = utils.MODELS / "als.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Train model first via `make train`.")
    with open(model_path, "rb") as f:
        meta = pickle.load(f)
    user_factors = np.load(utils.MODELS / "user_factors.npy")
    item_factors = np.load(utils.MODELS / "item_factors.npy")
    return user_factors, item_factors, meta


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(utils.PROC / "train.parquet")
    val = pd.read_parquet(utils.PROC / "val.parquet")
    test = pd.read_parquet(utils.PROC / "test.parquet")
    return train, val, test


def build_user_item_dict(df: pd.DataFrame) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = {}
    for row in df.itertuples():
        mapping.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return mapping


def evaluate(user_factors: np.ndarray, item_factors: np.ndarray, test_truth: dict[int, set[int]], seen: dict[int, set[int]], k: int) -> tuple[dict[str, float], dict[int, list[int]]]:
    precisions = []
    recalls = []
    ndcgs = []
    user_recs: dict[int, list[int]] = {}

    for user, positives in test_truth.items():
        if not positives or user >= user_factors.shape[0]:
            continue
        seen_items = seen.get(user, set())
        user_vec = user_factors[user]
        scores = item_factors @ user_vec
        if seen_items:
            scores[list(seen_items)] = -np.inf
        k_eff = min(k, scores.shape[0])
        top_k_idx = np.argpartition(scores, -k_eff)[-k_eff:]
        top_k = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        user_recs[user] = top_k.tolist()

        hits = [1 if item in positives else 0 for item in top_k]
        hit_count = sum(hits)
        precisions.append(hit_count / k_eff if k_eff else 0.0)
        recalls.append(hit_count / len(positives))

        dcg = 0.0
        for rank, item in enumerate(top_k, start=1):
            if item in positives:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_hits = min(len(positives), k)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return (
        {
            "p10": float(np.mean(precisions)) if precisions else 0.0,
            "r10": float(np.mean(recalls)) if recalls else 0.0,
            "ndcg10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        },
        user_recs,
    )


def update_metrics(metrics: dict[str, float]) -> None:
    metrics_path = utils.METRICS / "metrics.json"
    payload = {}
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.update({"p10": metrics["p10"], "r10": metrics["r10"], "ndcg10": metrics["ndcg10"]})
    utils.save_json(payload, metrics_path)


def main(args: argparse.Namespace) -> None:
    utils.ensure_dirs()
    user_factors, item_factors, _ = load_artifacts()
    train, val, test = prepare_datasets()
    seen = build_user_item_dict(pd.concat([train, val], ignore_index=True))
    truth = build_user_item_dict(test)
    metrics, _ = evaluate(user_factors, item_factors, truth, seen, args.k)
    update_metrics(metrics)


if __name__ == "__main__":
    main(parse_args())
