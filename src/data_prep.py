"""Data preparation and exploratory analysis pipeline."""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import utils

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from wordcloud import STOPWORDS, WordCloud
except Exception:
    WordCloud, STOPWORDS = None, None


EDA_FIGS = {
    # existing
    "ratings_hist": "ratings_hist.png",
    "top_genres": "top_genres_bar.png",
    "heatmap": "user_item_heatmap_sample.png",
    # new
    "avg_rating_hist": "average_rating_hist.png",
    "avg_rating_pie": "average_rating_pie.png",
    "books_per_year": "books_published_per_year.png",
    "avg_rating_by_year": "average_rating_by_year.png",
    "lang_counts": "books_by_language.png",
    "lang_counts_non_en": "books_by_language_non_en.png",
    "top10_most_rated_avg": "avg_rating_top10_most_rated.png",
    "top10_most_rated_dist": "ratings_dist_top10_most_rated.png",
    "cold_top10_dist": "ratings_dist_top10_high_rating_low_count.png",
    "titles_wordcloud": "titles_wordcloud.png",
}


def has_columns(df: pd.DataFrame, cols: list[str], fig_key: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[skip] {fig_key}: missing columns {missing}")
        return False
    return True


def assert_valid_image(path: Path, min_bytes: int = 2048, min_unique: int = 16) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.stat().st_size < min_bytes:
        raise ValueError(f"Image too small: {path} ({path.stat().st_size} bytes)")
    if Image is not None:
        try:
            im = Image.open(path).convert("RGB")
            arr = np.asarray(im)
            flat = arr.reshape(-1, arr.shape[-1])
            if flat.shape[0] > 250_000:
                rng = np.random.default_rng(0)
                idx = rng.choice(flat.shape[0], size=250_000, replace=False)
                flat = flat[idx]
            uniq = np.unique(flat, axis=0)
            if uniq.shape[0] < min_unique:
                raise ValueError(
                    f"Image looks blank/low-detail: {path} (unique colors={uniq.shape[0]})"
                )
        except Exception as e:
            warnings.warn(f"Validation warning for {path}: {e}")
    else:
        warnings.warn("Pillow not available; basic file-size check only.")


def savefig_checked(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    assert_valid_image(output)


def _update_metadata(key: str, payload: dict) -> None:
    meta_path = utils.PROC / "eda_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except json.JSONDecodeError:
        existing = {}
    existing[key] = payload
    meta_path.write_text(json.dumps(existing, indent=2, sort_keys=True))


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
    key = "ratings_hist"
    if not has_columns(ratings, ["rating"], key):
        return
    plt.figure(figsize=(8, 5))
    plt.hist(ratings["rating"], bins=np.arange(0.5, 5.5, 0.5), color="#3E64FF", edgecolor="white")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    savefig_checked(output)


def generate_top_genres(books: pd.DataFrame, output: Path) -> None:
    key = "top_genres"
    if not has_columns(books, ["primary_genre"], key):
        return
    plt.figure(figsize=(10, 6))
    counts = books["primary_genre"].value_counts().head(10)
    counts.sort_values().plot(kind="barh", color="#FF8C42")
    plt.title("Top 10 Genres (language proxy)")
    plt.xlabel("Number of Books")
    plt.ylabel("Genre")
    savefig_checked(output)


def generate_heatmap(ratings: pd.DataFrame, output: Path, n_users: int = 25, n_items: int = 25) -> None:
    key = "heatmap"
    if not has_columns(ratings, ["user_idx", "item_idx", "rating"], key):
        return
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
    savefig_checked(output)


def generate_average_rating_hist(books: pd.DataFrame, output: Path) -> None:
    key = "avg_rating_hist"
    if not has_columns(books, ["average_rating"], key):
        return
    plt.figure(figsize=(8, 5))
    vals = books["average_rating"].dropna().values
    if vals.size == 0:
        print(f"[skip] {key}: no ratings available")
        return
    bins = np.linspace(1.0, 5.0, 21)
    plt.hist(vals, bins=bins, color="#5DADE2", edgecolor="white")
    plt.title("Average Rating Distribution")
    plt.xlabel("Average Rating")
    plt.ylabel("Number of Books")
    savefig_checked(output)


def generate_average_rating_pie(books: pd.DataFrame, output: Path) -> None:
    key = "avg_rating_pie"
    if not has_columns(books, ["average_rating"], key):
        return
    bins = np.arange(1.0, 5.0 + 0.5, 0.5)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    rating_bin = pd.cut(books["average_rating"], bins=bins, labels=labels, include_lowest=True)
    rating_str = rating_bin.astype(str).dropna()
    counts = Counter([val for val in rating_str if val != "nan"])
    if not counts:
        print(f"[skip] {key}: insufficient rating data")
        return
    ordered_counts = pd.Series(counts).reindex(labels).fillna(0)
    pct = ordered_counts / ordered_counts.sum() * 100.0
    plt.figure(figsize=(10, 8))
    wedges, _ = plt.pie(pct.values, labels=None, startangle=140, wedgeprops={"edgecolor": "white"})
    plt.title("Percentage of Books by Average Rating")
    legend_labels = [f"{lab}: {p:.1f}%" for lab, p in zip(pct.index, pct.values)]
    plt.legend(wedges, legend_labels, title="Rating Range", loc="center left", bbox_to_anchor=(1, 0.5))
    savefig_checked(output)


def generate_books_published_per_year(books: pd.DataFrame, output: Path) -> None:
    key = "books_per_year"
    if not has_columns(books, ["original_publication_year"], key):
        return
    df = books.copy()
    df = df[pd.to_numeric(df["original_publication_year"], errors="coerce").notna()]
    df["original_publication_year"] = df["original_publication_year"].astype(int)
    df = df[df["original_publication_year"] >= 1500]
    if df.empty:
        print(f"[skip] {key}: no rows after year filter")
        return
    counts = df["original_publication_year"].value_counts().sort_index()
    plt.figure(figsize=(12, 5))
    plt.plot(counts.index, counts.values, color="#1ABC9C")
    plt.title("Number of Books Published per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Books")
    savefig_checked(output)
    _update_metadata(key, {"year_min": int(counts.index.min()), "year_max": int(counts.index.max())})


def generate_avg_rating_by_pub_year(books: pd.DataFrame, output: Path) -> None:
    key = "avg_rating_by_year"
    need = ["original_publication_year", "average_rating"]
    if not has_columns(books, need, key):
        return
    df = books.copy()
    df = df[pd.to_numeric(df["original_publication_year"], errors="coerce").notna()]
    df["original_publication_year"] = df["original_publication_year"].astype(int)
    df = df[df["original_publication_year"] >= 1500]
    if df.empty:
        print(f"[skip] {key}: no rows after year filter")
        return
    avg = df.groupby("original_publication_year")["average_rating"].mean()
    plt.figure(figsize=(12, 5))
    plt.plot(avg.index, avg.values, color="#8E44AD")
    plt.title("Average Rating by Publication Year")
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    savefig_checked(output)


def generate_language_counts(books: pd.DataFrame, output: Path) -> None:
    key = "lang_counts"
    if not has_columns(books, ["language_code"], key):
        return
    counts = books["language_code"].fillna("unknown").value_counts()
    plt.figure(figsize=(10, 8))
    plt.barh(counts.index[::-1], counts.values[::-1], color="#F4D03F")
    plt.title("Number of Books by Language")
    plt.xlabel("Number of Books")
    plt.ylabel("Language Code")
    savefig_checked(output)


def generate_language_counts_non_en(books: pd.DataFrame, output: Path) -> None:
    key = "lang_counts_non_en"
    if not has_columns(books, ["language_code"], key):
        return
    non_en = books[~books["language_code"].fillna("").str.contains("en", na=False)]
    if non_en.empty:
        print(f"[skip] {key}: no non-English rows")
        return
    counts = non_en["language_code"].value_counts()
    plt.figure(figsize=(10, 8))
    plt.barh(counts.index[::-1], counts.values[::-1], color="#E67E22")
    plt.title("Number of Books by Non-English Languages")
    plt.xlabel("Number of Books")
    plt.ylabel("Language Code")
    savefig_checked(output)


def _top10_most_rated(books: pd.DataFrame) -> pd.DataFrame:
    need = ["title", "ratings_count", "average_rating"]
    if not all(c in books.columns for c in need):
        return pd.DataFrame()
    df = books.dropna(subset=["ratings_count"]).copy()
    return df.sort_values(by="ratings_count", ascending=False).head(10)


def generate_top10_most_rated_avg_rating(books: pd.DataFrame, output: Path) -> None:
    key = "top10_most_rated_avg"
    top = _top10_most_rated(books)
    if top.empty:
        print(f"[skip] {key}: cannot compute top-10")
        return
    top = top.copy()
    plt.figure(figsize=(12, 6))
    plt.bar(top["title"], top["average_rating"], color="#3498DB")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Rating")
    plt.title("Average Rating of Top 10 Most Rated Books")
    savefig_checked(output)
    output_csv = utils.PROC / "top10_most_rated.csv"
    top[["title", "ratings_count", "average_rating"]].to_csv(output_csv, index=False)
    _update_metadata(
        key,
        {
            "csv": str(output_csv),
            "top_titles": top["title"].head(5).tolist(),
        },
    )


def generate_top10_most_rated_ratings_dist(books: pd.DataFrame, output: Path) -> None:
    key = "top10_most_rated_dist"
    need = [
        "title",
        "ratings_count",
        "ratings_1",
        "ratings_2",
        "ratings_3",
        "ratings_4",
        "ratings_5",
    ]
    if not has_columns(books, need, key):
        return
    top = _top10_most_rated(books)
    if top.empty:
        print(f"[skip] {key}: cannot compute top-10")
        return
    ratings_cols = ["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]
    tbl = top[["title"] + ratings_cols].set_index("title")
    pct = tbl.div(tbl.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct = pct.fillna(0)
    ax = pct.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
    ax.set_ylabel("Percentage of Ratings (%)")
    ax.set_title("Ratings 1~5 Distribution of Top 10 Most Rated Books")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Stars")
    savefig_checked(output)


def generate_cold_books_ratings_dist(books: pd.DataFrame, output: Path) -> None:
    key = "cold_top10_dist"
    need = [
        "title",
        "average_rating",
        "ratings_count",
        "ratings_1",
        "ratings_2",
        "ratings_3",
        "ratings_4",
        "ratings_5",
    ]
    if not has_columns(books, need, key):
        return
    high_rating_threshold = 4.0
    low_count_threshold = books["ratings_count"].dropna().quantile(0.20)
    cold = books[
        (books["average_rating"] >= high_rating_threshold)
        & (books["ratings_count"] <= low_count_threshold)
    ]
    if cold.empty:
        print(f"[skip] {key}: no cold books with given thresholds")
        return
    top_cold = cold.sort_values(by="average_rating", ascending=False).head(10)
    ratings_cols = ["ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]
    tbl = top_cold[["title"] + ratings_cols].set_index("title")
    pct = tbl.div(tbl.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct = pct.fillna(0)
    ax = pct.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="plasma")
    ax.set_ylabel("Percentage of Ratings (%)")
    ax.set_title("1~5 Star Rating Distribution of Top 10 High-Rating Low-Rating-Count Books")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Stars")
    savefig_checked(output)
    output_csv = utils.PROC / "top10_cold_books.csv"
    top_cold.to_csv(output_csv, index=False)
    _update_metadata(
        key,
        {
            "csv": str(output_csv),
            "thresholds": {
                "high_rating_threshold": high_rating_threshold,
                "low_count_threshold": float(low_count_threshold) if pd.notna(low_count_threshold) else None,
            },
        },
    )


def generate_titles_wordcloud(books: pd.DataFrame, output: Path) -> None:
    key = "titles_wordcloud"
    if WordCloud is None:
        print(f"[skip] {key}: wordcloud library not installed")
        return
    if not has_columns(books, ["title"], key):
        return
    titles = books["title"].dropna().astype(str).tolist()
    text = " ".join([t for t in titles if t.strip()])
    if not text:
        print(f"[skip] {key}: no titles to render")
        return
    stop = set(STOPWORDS) if STOPWORDS is not None else set()
    stop.update({"The", "And", "Of", "A", "In", "Book", "Volume", "Edition"})
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=stop,
        colormap="viridis",
        max_words=200,
    ).generate(text)
    plt.figure(figsize=(15, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Book Titles")
    savefig_checked(output)
    tokens = [token.lower() for title in titles for token in title.split()]
    filtered_tokens = [tok for tok in tokens if tok and tok[0].isalpha() and tok.title() not in stop]
    word_counts = Counter(filtered_tokens)
    _update_metadata(key, {"top_words": word_counts.most_common(15)})


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
    generate_average_rating_hist(books, utils.FIGS / EDA_FIGS["avg_rating_hist"])
    generate_average_rating_pie(books, utils.FIGS / EDA_FIGS["avg_rating_pie"])
    generate_books_published_per_year(books, utils.FIGS / EDA_FIGS["books_per_year"])
    generate_avg_rating_by_pub_year(books, utils.FIGS / EDA_FIGS["avg_rating_by_year"])
    generate_language_counts(books, utils.FIGS / EDA_FIGS["lang_counts"])
    generate_language_counts_non_en(books, utils.FIGS / EDA_FIGS["lang_counts_non_en"])
    generate_top10_most_rated_avg_rating(books, utils.FIGS / EDA_FIGS["top10_most_rated_avg"])
    generate_top10_most_rated_ratings_dist(books, utils.FIGS / EDA_FIGS["top10_most_rated_dist"])
    generate_cold_books_ratings_dist(books, utils.FIGS / EDA_FIGS["cold_top10_dist"])
    generate_titles_wordcloud(books, utils.FIGS / EDA_FIGS["titles_wordcloud"])

    stats = utils.describe_sparse_matrix(ratings["user_id"], ratings["book_id"])
    stats.to_csv(utils.PROC / "interaction_stats.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed)
