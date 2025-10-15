"""Data preparation and exploratory analysis pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import utils

EDA_FIGS = {
    "ratings_hist": "ratings_hist.png",
    "top_genres": "top_genres_bar.png",
    "heatmap": "user_item_heatmap_sample.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data and generate EDA artifacts.")
    parser.add_argument("--seed", type=int, default=utils.SEED, help="Random seed used for shuffling splits.")
    return parser.parse_args()


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    books = utils.read_csv(utils.RAW / "books.csv")
    ratings = utils.read_csv(utils.RAW / "ratings.csv")
    return books, ratings


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    books["language_code"] = books["language_code"].fillna("unknown")
    books["authors"] = books["authors"].fillna("Unknown")
    books["primary_genre"] = books["language_code"].str.upper()
    return books


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")
    return ratings


def attach_indices(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings["user_idx"] = ratings["user_id"].astype("category").cat.codes
    ratings["item_idx"] = ratings["book_id"].astype("category").cat.codes
    return ratings


def build_user_lookup(ratings: pd.DataFrame) -> pd.DataFrame:
    grouped = ratings.groupby("user_id")["rating"]
    users = grouped.agg(n_ratings="count", mean_rating="mean").reset_index()
    index_map = ratings.drop_duplicates("user_id")[["user_id", "user_idx"]]
    users = users.merge(index_map, on="user_id", how="left")
    users["region"] = (users["user_id"] % 3).map({0: "AMER", 1: "EMEA", 2: "APAC"})
    return users.rename(columns={"user_idx": "user_idx"})


def build_book_lookup(books: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    index_map = ratings.drop_duplicates("book_id")[["book_id", "item_idx"]]
    base_cols = [
        "book_id",
        "title",
        "authors",
        "language_code",
        "primary_genre",
        "average_rating",
        "ratings_count",
    ]
    available_cols = [c for c in base_cols if c in books.columns]
    book_lookup = books[available_cols].copy()
    book_lookup = book_lookup.merge(index_map, on="book_id", how="inner")
    return book_lookup


def generate_ratings_hist(ratings: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(ratings["rating"], bins=np.arange(0.5, 5.5, 0.5), color="#3E64FF", edgecolor="white")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def generate_top_genres(books: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    counts = books["primary_genre"].value_counts().head(10)
    counts.sort_values().plot(kind="barh", color="#FF8C42")
    plt.title("Top 10 Genres (language proxy)")
    plt.xlabel("Number of Books")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def generate_heatmap(ratings: pd.DataFrame, output: Path, n_users: int = 25, n_items: int = 25) -> None:
    subset = ratings.sample(n=min(len(ratings), n_users * n_items), random_state=0)
    pivot = subset.pivot_table(
        index="user_idx",
        columns="item_idx",
        values="rating",
        aggfunc="mean",
    )
    pivot = pivot.fillna(0)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Rating")
    plt.title("User-Item Rating Heatmap (sample)")
    plt.xlabel("Item Index")
    plt.ylabel("User Index")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def save_partitions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, df in {"train": train, "val": val, "test": test}.items():
        df.to_parquet(utils.PROC / f"{name}.parquet", index=False)


def main(seed: int) -> None:
    utils.ensure_dirs()
    utils.set_seed(seed)

    books_raw, ratings_raw = load_sources()
    books = clean_books(books_raw)
    ratings = clean_ratings(ratings_raw)
    ratings = attach_indices(ratings)

    train, val, test = utils.train_val_test_split(ratings, seed=seed)
    save_partitions(train, val, test)

    books_lookup = build_book_lookup(books, ratings)
    users_lookup = build_user_lookup(ratings)
    books_lookup.to_csv(utils.PROC / "books.csv", index=False)
    users_lookup.to_csv(utils.PROC / "users.csv", index=False)

    generate_ratings_hist(ratings, utils.FIGS / EDA_FIGS["ratings_hist"])
    generate_top_genres(books, utils.FIGS / EDA_FIGS["top_genres"])
    generate_heatmap(ratings, utils.FIGS / EDA_FIGS["heatmap"])

    stats = utils.describe_sparse_matrix(ratings["user_id"], ratings["book_id"])
    stats.to_csv(utils.PROC / "interaction_stats.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed)
