"""ALS model training with simple hyperparameter search."""
from __future__ import annotations

import argparse
import itertools
import json
import pickle

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from . import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ALS recommender with hyperparameter sweep.")
    parser.add_argument("--seed", type=int, default=utils.SEED, help="Random seed for reproducibility.")
    parser.add_argument("--rank", nargs="+", type=int, default=[64], help="List of latent factors to try.")
    parser.add_argument("--reg", nargs="+", type=float, default=[0.05], help="List of regularization values.")
    parser.add_argument("--alpha", nargs="+", type=float, default=[20.0], help="List of alpha confidence scalars.")
    parser.add_argument("--iters", nargs="+", type=int, default=[10], help="List of iteration counts.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for ranking metrics.")
    return parser.parse_args()


def load_split(name: str) -> pd.DataFrame:
    path = utils.PROC / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected split parquet at {path}")
    return pd.read_parquet(path)


def build_matrix(df: pd.DataFrame, n_users: int, n_items: int, alpha: float) -> sparse.coo_matrix:
    data = df["rating"].astype(float).values * alpha
    rows = df["item_idx"].astype(int).values
    cols = df["user_idx"].astype(int).values
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_users))


def ndcg_at_k(model: AlternatingLeastSquares, interactions: sparse.csr_matrix, ground_truth: dict[int, set[int]], k: int) -> float:
    ndcgs = []
    for user, positives in ground_truth.items():
        if not positives:
            continue
        recs, _ = model.recommend(user, interactions, N=k, filter_already_liked_items=False)
        dcg = 0.0
        for rank, item in enumerate(recs, start=1):
            if item in positives:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_hits = min(len(positives), k)
        if ideal_hits == 0:
            continue
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def prepare_ground_truth(
    df: pd.DataFrame, allowed_users: set[int] | None = None
) -> dict[int, set[int]]:
    """Collect validation items keyed by user, optionally filtering users.

    ALS can only score users that were present during training. Some validation
    rows may belong to cold-start users which would otherwise raise an index
    error when we ask the model for recommendations. By skipping those users we
    evaluate the model only where factors exist.
    """

    truth: dict[int, set[int]] = {}
    for row in df.itertuples():
        user = int(row.user_idx)
        if allowed_users is not None and user not in allowed_users:
            continue
        truth.setdefault(user, set()).add(int(row.item_idx))
    return truth


def sweep(train_df: pd.DataFrame, val_df: pd.DataFrame, args: argparse.Namespace) -> tuple[AlternatingLeastSquares, dict[str, float]]:
    n_users = int(max(train_df.user_idx.max(), val_df.user_idx.max()) + 1)
    n_items = int(max(train_df.item_idx.max(), val_df.item_idx.max()) + 1)

    interactions = build_matrix(train_df, n_users, n_items, alpha=1.0)
    user_items = interactions.T.tocsr()
    train_users = set(train_df["user_idx"].astype(int).unique())
    ground_truth = prepare_ground_truth(val_df, allowed_users=train_users)

    best_score = -np.inf
    best_model: AlternatingLeastSquares | None = None
    best_params: dict[str, float] = {}

    for rank, reg, alpha, iters in itertools.product(args.rank, args.reg, args.alpha, args.iters):
        model = AlternatingLeastSquares(
            factors=rank,
            regularization=reg,
            iterations=iters,
            random_state=args.seed,
        )
        model.fit(build_matrix(train_df, n_users, n_items, alpha=alpha))
        score = ndcg_at_k(model, user_items, ground_truth, k=args.k)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = {
                "rank": rank,
                "reg": reg,
                "alpha": alpha,
                "iters": iters,
                "ndcg10_val": score,
            }
    if best_model is None:
        raise RuntimeError("Hyperparameter sweep failed to produce a model.")
    return best_model, best_params


def retrain_full(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict[str, float],
    seed: int,
) -> tuple[AlternatingLeastSquares, sparse.coo_matrix]:
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    n_users = int(full_df.user_idx.max() + 1)
    n_items = int(full_df.item_idx.max() + 1)
    matrix = build_matrix(full_df, n_users, n_items, alpha=float(params["alpha"]))
    model = AlternatingLeastSquares(
        factors=int(params["rank"]),
        regularization=float(params["reg"]),
        iterations=int(params["iters"]),
        random_state=seed,
    )
    model.fit(matrix)
    return model, matrix


def persist_model(model: AlternatingLeastSquares, params: dict[str, float], matrix: sparse.coo_matrix) -> None:
    utils.ensure_dirs()
    models_path = utils.MODELS
    models_path.mkdir(parents=True, exist_ok=True)

    np.save(models_path / "user_factors.npy", model.user_factors)
    np.save(models_path / "item_factors.npy", model.item_factors)

    payload = {
        "params": params,
        "n_users": int(matrix.shape[1]),
        "n_items": int(matrix.shape[0]),
        "alpha": float(params.get("alpha", 1.0)),
    }
    with open(models_path / "als.pkl", "wb") as f:
        pickle.dump(payload, f)


def update_metrics(params: dict[str, float]) -> None:
    metrics_path = utils.METRICS / "metrics.json"
    existing: dict[str, float] = {}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing.setdefault("params", {}).update({k: float(v) for k, v in params.items() if k != "ndcg10_val"})
    existing.setdefault("validation", {})["ndcg10"] = float(params.get("ndcg10_val", 0.0))
    utils.save_json(existing, metrics_path)


def main(args: argparse.Namespace) -> None:
    utils.ensure_dirs()
    utils.set_seed(args.seed)

    train_df = load_split("train")
    val_df = load_split("val")

    best_model, params = sweep(train_df, val_df, args)
    final_model, matrix = retrain_full(train_df, val_df, params, args.seed)
    persist_model(final_model, params, matrix)
    update_metrics(params)


if __name__ == "__main__":
    main(parse_args())
