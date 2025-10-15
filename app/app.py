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
        books = books.set_index("item_idx", drop=False)
    if not users.empty and "user_idx" in users.columns:
        users = users.set_index("user_idx", drop=False)
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
    metrics_path = METRICS / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


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
        recs = recs.merge(books, on="item_idx", how="left")
        if genre and genre != "All":
            recs = recs[recs["primary_genre"].fillna("Unknown") == genre]
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

st.title("📚 BookTrends Recommendation System")
st.markdown("Discover personalized book suggestions while monitoring quality and fairness.")

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

st.subheader("Evaluation Metrics")
if metrics:
    metrics_table = {k: v for k, v in metrics.items() if k in {"p10", "r10", "ndcg10"}}
    st.table(pd.DataFrame([metrics_table]))
else:
    st.info("Run `make eval` to compute evaluation metrics.")

st.subheader("Fairness Snapshot")
if not parity_df.empty:
    st.dataframe(parity_df)
else:
    st.info("Run `make parity` to compute parity metrics.")
