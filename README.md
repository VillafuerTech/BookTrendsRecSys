# BookTrends RecSys

BookTrends RecSys is a lightweight end-to-end recommender system pipeline that prepares Goodreads-style data, trains an implicit Alternating Least Squares (ALS) model, evaluates ranking quality, and surfaces a Streamlit demo with fairness diagnostics. The project is organized for production workflows with reproducible Makefile targets and artifact tracking.

## Quickstart

1. `make setup`
2. Place `books.csv` and `ratings.csv` under `data/raw/`
3. `make data`
4. `make train && make eval && make parity`
5. `make app`

## Data Collection

Use the Goodreads crawlers to bootstrap training data from public review
pages:

1. Populate `src/ingest/assets/seed_books.csv` with a header row that
   includes `book_id` (optionally add `title`).
2. `make crawl_books` downloads reviews into
   `data/raw/reviews/goodreads_reviews_<book_id>.json` while respecting
   retries, pagination and resume support.
3. `make crawl_users` loads the collected reviews, identifies unique
   reviewers, and fetches their rating history into
   `data/raw/users/goodreads_ratings_<user_id>.json`.
4. `make count_books` builds coverage metrics in
   `data/interim/book_counts.csv` (columns: `book_slug,count`).
5. `make build_dataset` merges all user histories into
   `data/processed/goodreads_dataset.csv` with columns
   `userid,bookid,rating` sorted deterministically.

Each script logs progress to `logs/collect.log` and stdout, supports
configurable delays and user agents, and exits cleanly when run on empty
directories for dry-run validation.

## Metrics Snapshot

The latest ranking metrics are written to `metrics/metrics.json`. After running `make eval` you can inspect results via:

```bash
python -c "import json;print(json.dumps(json.load(open('metrics/metrics.json')), indent=2))"
```

This file stores:

- `p10`: Precision@10 on the held-out test split.
- `r10`: Recall@10 on the held-out test split.
- `ndcg10`: NDCG@10 on the held-out test split.
- `params`: Tuned hyperparameters selected during `make train`.

## Fairness & Parity

`make parity` produces `metrics/parity.csv`, reporting per-group ranking performance for user regions (synthetic AMER/EMEA/APAC buckets) and language-based genre proxies. The CSV also includes `gap` rows capturing the max-min difference for each metric. Monitor these values over time to detect regressions in parity.

## Project Layout

```
.
├── app/                 # Streamlit demo
├── data/                # Raw and processed datasets
├── deliverables/        # Presentation scaffold
├── figs/eda/            # EDA visualizations
├── metrics/             # Evaluation and parity artifacts
├── models/              # Saved ALS factors
├── src/                 # Data prep, training, evaluation, parity utilities
└── Makefile             # Reproducible entrypoints
```

## Seeds, Versions & License

- Default random seed: `42` (override with `SEED` make variable).
- Python dependencies are listed in `requirements.txt`.
- Licensed under the terms described in `LICENSE`.
