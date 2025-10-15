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


def humanize_pct(x: float) -> str:
    try:
        if x is None:
            return "–"
        if x < 0.001:
            return "<0.1%"
        return f"{x:.1%}"
    except Exception:
        return "–"


def load_metrics_json() -> dict:
    p = Path("metrics/metrics.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def compute_parity_gap(csv_path="metrics/parity.csv", metric="ndcg10") -> str:
    p = Path(csv_path)
    if not p.exists():
        return "–"
    try:
        df = pd.read_csv(p)
        df_m = df[df["metric"] == metric]
        if df_m.empty:
            return "–"
        gap = df_m["value"].max() - df_m["value"].min()
        return humanize_pct(gap)
    except Exception:
        return "–"


def render_help_sidebar(metrics: dict, parity_gap: str):
    p10 = humanize_pct(metrics.get("p10")) if metrics else "–"
    r10 = humanize_pct(metrics.get("r10")) if metrics else "–"
    ndcg10 = humanize_pct(metrics.get("ndcg10")) if metrics else "–"

    st.sidebar.markdown("### ℹ️ About & How to read this page")
    st.sidebar.markdown(
        """
**BookTrends** suggests books you may like **now**.  
We show a short list (Top-10) and a couple of simple scores so you can trust the list.  
Higher scores are better, and a **small fairness gap** means different groups get similar quality.
        """
    )
    st.sidebar.caption(
        f"Today’s snapshot — P@10: **{p10}**, R@10: **{r10}**, NDCG@10: **{ndcg10}**, Parity gap: **{parity_gap}**"
    )


def render_explainers(metrics: dict):
    st.divider()
    with st.expander("📊 What do these metrics mean?"):
        st.markdown(
            """
- **Precision@10 (P@10)** — *Out of the 10 suggestions we show, how many were actually good fits for you?*  
  Example: if 3 of the 10 were spot-on, **P@10 = 30%**.

- **Recall@10 (R@10)** — *Out of all the good books out there for you, how many did we manage to include in the Top-10?*  
  Example: if there are 20 books you'd love and we surfaced 4, **R@10 = 20%**.

- **NDCG@10** — *A quality score that also cares about **order**: great picks near the top boost the score more.*  
  It ranges from 0 to 100% (perfect ranking). Higher is better.
            """
        )
        if metrics:
            st.caption(
                f"Current model snapshot — P@10: **{humanize_pct(metrics.get('p10'))}**, "
                f"R@10: **{humanize_pct(metrics.get('r10'))}**, "
                f"NDCG@10: **{humanize_pct(metrics.get('ndcg10'))}**."
            )

    with st.expander("🎚️ Genres & filters"):
        st.markdown(
            """
- **Genre filter** narrows the list to a category (e.g., Fantasy, Romance).  
- Choosing **All** shows any genre. Picking a specific genre may return **fewer** items.
- If you see **“No recommendations available”**, try:
  1) switch back to **All**,  
  2) pick another genre, or  
  3) try a different user.
- We only show items we’re reasonably confident about after filtering.
            """
        )

    with st.expander("⚖️ Fairness snapshot"):
        st.markdown(
            """
This table compares the same metric across groups (like **regions** or **genres**).  
A **small gap** between groups means the model treats groups similarly.  
A **large gap** suggests one group gets better/worse recommendations and needs attention.
            """
        )
        gap = compute_parity_gap(metric="ndcg10")
        if gap != "–":
            st.caption(f"Current ndcg10 gap across groups: **{gap}**.")
        else:
            st.caption("Current fairness gap data unavailable.")

st.set_page_config(page_title="BookTrends Recommender", layout="wide")
utils.ensure_dirs()

MODELS = utils.MODELS
METRICS = utils.METRICS
PROC = utils.PROC


@st.cache_data
def load_lookup_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    books = pd.read_csv(PROC / "books.csv") if (PROC / "books.csv").exists() else pd.DataFrame()
    users = pd.read_csv(PROC / "users.csv") if (PROC / "users.csv").exists() else pd.DataFrame()
    if not books.empty and "item_idx" in books.columns:
        books = books.set_index("item_idx")
    if not users.empty and "user_idx" in users.columns:
        users = users.set_index("user_idx")
    return books, users


@st.cache_resource
def load_model() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model_path = MODELS / "als.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Train model first with `make train`.")
    with open(MODELS / "als.pkl", "rb") as f:
        meta = pickle.load(f)
    user_factors = np.load(MODELS / "user_factors.npy")
    item_factors = np.load(MODELS / "item_factors.npy")
    return user_factors, item_factors, meta


@st.cache_data
def load_metrics() -> dict:
    return load_metrics_json()


@st.cache_data
def load_parity() -> pd.DataFrame:
    parity_path = METRICS / "parity.csv"
    if parity_path.exists():
        return pd.read_csv(parity_path)
    return pd.DataFrame(columns=["metric", "group", "group_value", "value"])


def recommend(
    user_identifier: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    books: pd.DataFrame,
    users: pd.DataFrame,
    genre: Optional[str],
    k: int = 10,
) -> pd.DataFrame:
    if user_identifier >= user_factors.shape[0]:
        st.warning("User index outside factor matrix; defaulting to top books overall.")
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


books_df, users_df = load_lookup_tables()
try:
    user_factors, item_factors, meta = load_model()
except FileNotFoundError:
    st.error("Model artifacts missing. Run `make train` then `make eval` and `make parity`.")
    st.stop()
metrics = load_metrics()
parity_df = load_parity()
parity_gap = compute_parity_gap(metric="ndcg10")

render_help_sidebar(metrics, parity_gap)

st.title("📚 BookTrends Recommendation System")
st.markdown("Discover personalized book suggestions while monitoring quality and fairness.")

render_explainers(metrics)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    user_input = st.number_input("User index", min_value=0, max_value=int(user_factors.shape[0] - 1), value=0, step=1)
with col2:
    genre_options = ["All"] + sorted(books_df["primary_genre"].dropna().unique().tolist()) if not books_df.empty else ["All"]
    genre_choice = st.selectbox("Genre filter", genre_options)
with col3:
    st.write("**Model hyperparameters**")
    st.json(meta.get("params", {}))

if st.button("Recommend"):
    results = recommend(user_input, user_factors, item_factors, books_df, users_df, genre_choice)
    if results.empty:
        st.info("No recommendations available for the selected filters.")
    else:
        display_cols = [
            col for col in ["title", "authors", "primary_genre", "average_rating", "ratings_count", "score"] if col in results
        ]
        st.dataframe(results[display_cols].reset_index(drop=True))

st.subheader("Evaluation Metrics 🛈")
st.caption("🛈 Need a refresher? Open the “What do these metrics mean?” panel above.")
if metrics:
    metrics_table = {
        "P@10": humanize_pct(metrics.get("p10")),
        "R@10": humanize_pct(metrics.get("r10")),
        "NDCG@10": humanize_pct(metrics.get("ndcg10")),
    }
    st.table(pd.DataFrame([metrics_table]))
else:
    st.info("Run `make eval` to compute evaluation metrics.")

st.subheader("Fairness Snapshot 🛈")
st.caption("🛈 Curious about the gap? Expand “Fairness snapshot” above for context.")
if not parity_df.empty:
    st.dataframe(parity_df)
else:
    st.info("Run `make parity` to compute parity metrics.")
