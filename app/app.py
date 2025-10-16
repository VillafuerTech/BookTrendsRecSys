"""Streamlit interface for BookTrends recommender."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import utils

FIGS = Path("figs/eda")
DATA_PROC = Path("data/processed")
METRICS_DIR = Path("metrics")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def humanize_pct(x) -> str:
    try:
        if x is None:
            return "–"
        if x < 0.001:
            return "<0.1%"
        return f"{float(x):.1%}"
    except Exception:
        return "–"


def humanize_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "–"


def show_image_card(filename: str, title: str, caption: str):
    path = FIGS / filename
    if path.exists():
        st.markdown(f"**{title}**")
        st.image(str(path), use_column_width=True, caption=caption)
    else:
        st.caption(f"[skip] Missing figure: {filename}")


def load_metrics_snapshot() -> dict:
    @st.cache_data(show_spinner=False)
    def _load() -> dict:
        m = load_json(METRICS_DIR / "metrics.json")
        if "params" in m:
            m.pop("params", None)
        return m

    return _load()


def parity_gap_from_csv(metric_name: str = "ndcg10") -> tuple[pd.DataFrame, str]:
    df = load_csv(METRICS_DIR / "parity.csv")
    if df.empty or "metric" not in df.columns:
        return pd.DataFrame(), "–"
    dfm = df[df["metric"] == metric_name].copy()
    if dfm.empty:
        return pd.DataFrame(), "–"
    try:
        gap = dfm["value"].max() - dfm["value"].min()
        gap_txt = humanize_pct(gap) if gap is not None else "–"
    except Exception:
        gap_txt = "–"
    return dfm, gap_txt


@st.cache_data(show_spinner=False)
def load_lookup_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    books = load_csv(DATA_PROC / "books.csv")
    users = load_csv(DATA_PROC / "users.csv")
    if not books.empty and "item_idx" in books.columns:
        books = books.set_index("item_idx")
    if not users.empty and "user_idx" in users.columns:
        users = users.set_index("user_idx")
    return books, users


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model_path = utils.MODELS / "als.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model artifacts missing. Run `make train` then `make eval`.")
    with open(model_path, "rb") as f:
        meta = pickle.load(f)
    user_factors = np.load(utils.MODELS / "user_factors.npy")
    item_factors = np.load(utils.MODELS / "item_factors.npy")
    return user_factors, item_factors, meta


def recommend(
    user_identifier: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    books: pd.DataFrame,
    users: pd.DataFrame,
    genre: Optional[str],
    k: int = 10,
) -> pd.DataFrame:
    if user_factors.size == 0 or item_factors.size == 0:
        return pd.DataFrame()
    if user_identifier >= user_factors.shape[0]:
        st.warning("User index outside factor matrix; showing popular picks instead.")
        scores = item_factors.mean(axis=1)
    else:
        scores = item_factors @ user_factors[user_identifier]
    k_eff = min(k, scores.shape[0])
    top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    recs = pd.DataFrame({"item_idx": top_idx, "score": scores[top_idx]})
    if not books.empty:
        enriched = recs.join(books, on="item_idx", how="left")
        if "title" in enriched.columns:
            enriched = enriched[enriched["title"].notna() & (enriched["title"].astype(str).str.strip() != "")]
        if genre and genre != "All" and "primary_genre" in enriched.columns:
            enriched = enriched[enriched["primary_genre"].fillna("Unknown") == genre]
        recs = enriched.reset_index(drop=True)
    if not users.empty and user_identifier in users.index:
        recs["user_region"] = users.loc[user_identifier, "region"]
    return recs.head(k)


def render_recommendations_tab(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    books_df: pd.DataFrame,
    users_df: pd.DataFrame,
):
    st.subheader("Get book ideas")
    st.caption("Pick a user or a book and (optionally) a genre. We’ll show a short list you can skim quickly.")

    if user_factors.size == 0 or item_factors.size == 0:
        st.info("Recommendations are unavailable because model files are missing.")
        return

    max_user_idx = int(user_factors.shape[0] - 1)
    col1, col2 = st.columns([1, 1])
    with col1:
        user_input = st.number_input(
            "User index",
            min_value=0,
            max_value=max(0, max_user_idx),
            value=0,
            step=1,
        )
    with col2:
        genre_options = ["All"]
        if not books_df.empty and "primary_genre" in books_df.columns:
            genre_options += sorted(
                {g for g in books_df["primary_genre"].dropna().astype(str).tolist() if g.strip()}
            )
        genre_choice = st.selectbox("Genre filter", genre_options)

    if st.button("Show recommendations", type="primary"):
        results = recommend(int(user_input), user_factors, item_factors, books_df, users_df, genre_choice)
        if results.empty:
            st.info("No recommendations available for the selected filters.")
        else:
            display_cols = [
                col
                for col in [
                    "title",
                    "authors",
                    "primary_genre",
                    "average_rating",
                    "ratings_count",
                    "score",
                    "user_region",
                ]
                if col in results
            ]
            st.dataframe(results[display_cols].reset_index(drop=True))

    metrics = load_metrics_snapshot()
    if metrics:
        st.divider()
        st.markdown("#### How good are the suggestions (overall)?")
        st.caption("What these numbers mean")
        p10 = humanize_pct(metrics.get("p10"))
        r10 = humanize_pct(metrics.get("r10"))
        ndcg = humanize_pct(metrics.get("ndcg10"))
        st.markdown(
            f"- **Precision@10:** {p10} — out of the 10 shown, the fraction that were good.\n"
            f"- **Recall@10:** {r10} — how much of the good stuff we surfaced in 10.\n"
            f"- **NDCG@10:** {ndcg} — overall list quality (higher is better)."
        )
    else:
        st.caption("Metrics not available yet. Run evaluation to generate them.")


def render_insights_tab():
    st.subheader("What the data looks like")

    eda_summary = load_json(DATA_PROC / "eda_summary.json")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Books", humanize_int(eda_summary.get("n_books")))
    with col2:
        st.metric("Users", humanize_int(eda_summary.get("n_users")))
    with col3:
        st.metric("Interactions", humanize_int(eda_summary.get("n_interactions")))
    with col4:
        st.metric("Sparsity", humanize_pct(eda_summary.get("sparsity")))

    st.divider()
    st.markdown("### Key patterns at a glance")

    c1, c2 = st.columns(2)
    with c1:
        show_image_card(
            "average_rating_hist.png",
            "How people rate books",
            "Most titles cluster around the mid–high range.",
        )
    with c2:
        show_image_card(
            "books_published_per_year.png",
            "Publishing over time",
            "How many books appear each year in the catalog.",
        )

    c3, c4 = st.columns(2)
    with c3:
        show_image_card(
            "rating_vs_count_scatter.png",
            "Ratings vs popularity",
            "Popular books (more ratings) aren’t always higher rated.",
        )
    with c4:
        show_image_card(
            "books_by_language.png",
            "Languages",
            "Top languages represented in the catalog.",
        )

    c5, c6 = st.columns(2)
    with c5:
        show_image_card(
            "ratings_dist_top10_most_rated.png",
            "Most-rated books",
            "Distribution of 1–5 star ratings for the 10 most-rated titles.",
        )
    with c6:
        show_image_card(
            "ratings_dist_top10_cold.png",
            "Hidden gems",
            "High-rated books with relatively few ratings.",
        )

    c7, c8 = st.columns(2)
    with c7:
        show_image_card(
            "long_tail_coverage.png",
            "The long tail",
            "A small set of items gets most of the attention.",
        )
    with c8:
        show_image_card(
            "user_item_heatmap_sample.png",
            "A peek at the matrix",
            "A tiny sample of the user × book interactions.",
        )

    st.markdown("### Extra slice")
    show_image_card(
        "books_by_language_non_en.png",
        "Non-English languages",
        "Breakdown outside of English.",
    )

    top10 = load_csv(DATA_PROC / "top10_most_rated.csv")
    cold10 = load_csv(DATA_PROC / "top10_cold_books.csv")
    if not top10.empty or not cold10.empty:
        st.divider()
        st.markdown("### Download quick lists")
        col_a, col_b = st.columns(2)
        if not top10.empty:
            col_a.dataframe(top10.head(10))
            col_a.download_button(
                "Download Top-10 Most-Rated (CSV)",
                data=top10.to_csv(index=False).encode("utf-8"),
                file_name="top10_most_rated.csv",
                mime="text/csv",
            )
        if not cold10.empty:
            col_b.dataframe(cold10.head(10))
            col_b.download_button(
                "Download Cold Gems (CSV)",
                data=cold10.to_csv(index=False).encode("utf-8"),
                file_name="top10_cold_books.csv",
                mime="text/csv",
            )

    st.divider()
    st.markdown("### Fairness snapshot")
    st.caption("We compare quality across groups (e.g., regions or genres). Smaller gap ≈ more similar quality.")
    df_parity, gap_txt = parity_gap_from_csv(metric_name="ndcg10")
    if not df_parity.empty:
        st.write(f"**Gap (NDCG@10):** {gap_txt}")
        st.dataframe(df_parity)
    else:
        st.caption("[skip] Parity file not found. Run evaluation to generate it.")


def main():
    st.set_page_config(page_title="BookTrendsRecSys", layout="wide")
    st.title("BookTrendsRecSys")
    st.caption("Find books you may enjoy. Friendly insights included.")

    utils.ensure_dirs()

    books_df, users_df = load_lookup_tables()
    try:
        user_factors, item_factors, _ = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    tab_rec, tab_ins = st.tabs(["Recommendations", "Insights"])
    with tab_rec:
        render_recommendations_tab(user_factors, item_factors, books_df, users_df)
    with tab_ins:
        render_insights_tab()


if __name__ == "__main__":
    main()

