"""Simple parity analysis across user and item strata."""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from . import utils
from .eval_ranking import (
    build_user_item_dict,
    evaluate,
    load_artifacts,
    prepare_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute parity metrics for ALS recommender.")
    parser.add_argument("--seed", type=int, default=utils.SEED)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--by", action="append", default=["region", "genre"], help="Grouping dimensions (repeatable).")
    return parser.parse_args()


def load_group_maps() -> tuple[dict[int, str], dict[int, str]]:
    users_path = utils.PROC / "users.csv"
    books_path = utils.PROC / "books.csv"
    if not users_path.exists() or not books_path.exists():
        raise FileNotFoundError("Run `make data` before parity analysis.")
    users = pd.read_csv(users_path)
    books = pd.read_csv(books_path)
    user_groups = dict(zip(users["user_idx"], users.get("region", "UNKNOWN")))
    item_groups = dict(zip(books["item_idx"], books.get("primary_genre", "UNKNOWN")))
    return user_groups, item_groups


def compute_single_metric(top_k: Iterable[int], positives: set[int], k: int) -> dict[str, float] | None:
    positives = set(positives)
    if not positives:
        return None
    top_list = list(top_k)
    k_eff = min(k, len(top_list))
    if k_eff == 0:
        return None
    hit_count = sum(1 for item in top_list[:k_eff] if item in positives)
    precision = hit_count / k_eff
    recall = hit_count / len(positives)
    dcg = 0.0
    for rank, item in enumerate(top_list[:k_eff], start=1):
        if item in positives:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(positives), k_eff)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {"p10": precision, "r10": recall, "ndcg10": ndcg}


def aggregate_metrics(per_group: dict[str, list[dict[str, float]]]) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for group_value, metrics_list in per_group.items():
        if not metrics_list:
            continue
        df = pd.DataFrame(metrics_list)
        for metric in ["p10", "r10", "ndcg10"]:
            rows.append((metric, group_value, float(df[metric].mean())))
    return rows


def compute_gaps(rows: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    df = pd.DataFrame(rows, columns=["metric", "group_value", "value"])
    gaps: list[tuple[str, str, float]] = []
    for metric, group_df in df.groupby("metric"):
        if group_df.empty:
            continue
        gap = group_df["value"].max() - group_df["value"].min()
        gaps.append((metric, "gap", float(gap)))
    return gaps


def main(args: argparse.Namespace) -> None:
    utils.ensure_dirs()
    user_factors, item_factors, _ = load_artifacts()
    train, val, test = prepare_datasets()
    seen = build_user_item_dict(pd.concat([train, val], ignore_index=True))
    truth = build_user_item_dict(test)
    metrics, user_recs = evaluate(user_factors, item_factors, truth, seen, args.k)

    user_groups, item_groups = load_group_maps()
    requested = []
    for entry in args.by:
        if isinstance(entry, (list, tuple)):
            requested.extend(entry)
        else:
            requested.append(entry)

    records: list[dict[str, str | float]] = []

    for group in requested:
        group = group.lower()
        if group == "region":
            per_group: dict[str, list[dict[str, float]]] = defaultdict(list)
            for user, positives in truth.items():
                top_k = user_recs.get(user)
                if top_k is None:
                    continue
                group_value = user_groups.get(user, "UNKNOWN")
                metric_values = compute_single_metric(top_k, positives, args.k)
                if metric_values:
                    per_group[group_value].append(metric_values)
            aggregated = aggregate_metrics(per_group)
            for metric, group_value, value in aggregated:
                records.append({"metric": metric, "group": "region", "group_value": group_value, "value": value})
            for metric, group_value, value in compute_gaps(aggregated):
                records.append({"metric": metric, "group": "region", "group_value": group_value, "value": value})
        elif group == "genre":
            per_group: dict[str, list[dict[str, float]]] = defaultdict(list)
            for user, positives in truth.items():
                top_k = user_recs.get(user)
                if top_k is None:
                    continue
                genre_buckets: dict[str, set[int]] = defaultdict(set)
                for item in positives:
                    genre = item_groups.get(item, "UNKNOWN")
                    genre_buckets[genre].add(item)
                for genre_value, genre_items in genre_buckets.items():
                    metric_values = compute_single_metric(top_k, genre_items, args.k)
                    if metric_values:
                        per_group[genre_value].append(metric_values)
            aggregated = aggregate_metrics(per_group)
            for metric, group_value, value in aggregated:
                records.append({"metric": metric, "group": "genre", "group_value": group_value, "value": value})
            for metric, group_value, value in compute_gaps(aggregated):
                records.append({"metric": metric, "group": "genre", "group_value": group_value, "value": value})

    df = pd.DataFrame(records)
    df.to_csv(utils.METRICS / "parity.csv", index=False)


if __name__ == "__main__":
    main(parse_args())
