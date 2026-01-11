# BookTrends RecSys

BookTrends RecSys is a lightweight, end-to-end recommender system pipeline designed to collect data, train an implicit Alternating Least Squares (ALS) model, evaluate ranking quality, and surface results via an interactive Streamlit application. It includes modules for fairness diagnostics and exploratory data analysis.

This project was developed as part of the **Introduction to Data Science** course at the University of Helsinki.

## Project Architecture

The system consists of four main stages:

1.  **Data Collection**: Custom scraping scripts to build a Goodreads-style dataset.
2.  **Data Preparation**: preprocessing of interactions and metadata.
3.  **Modeling & Evaluation**: Training an ALS model and evaluating it with ranking metrics and fairness checks.
4.  **Application**: A user-facing dashboard for recommendations and insights.

### 1. Data Collection (`scripts/`)
The project includes a suite of scripts to scrape data from Goodreads:
-   `get_rating_of_books.py`: Scrapes book metadata and user reviews for popular books.
-   `get_rating_of_users.py`: Fetches full rating histories for users identified in reviews.
-   `build_goodreads_dataset.py`: Merges raw JSON files into a structured `goodreads_dataset.csv`.
-   `count_books.py`: Analyzes book coverage to filter underrepresented titles.

### 2. Modeling (`src/`)
-   **Algorithm**: Implicit Alternating Least Squares (ALS).
-   **Training**: `src.als_train` trains the model using processed data.
-   **Artifacts**: User and Item factors are saved to `models/` for inference.

### 3. Metrics & Fairness (`metrics/`)
Evaluation is twofold:
-   **Ranking Metrics**: Precision@10, Recall@10, and NDCG@10 on a held-out test set. Results are stored in `metrics/metrics.json`.
-   **Parity Analysis**: `src.parity` calculates group-based performance metrics (e.g., by User Region or Book Language) to detect potential biases. Results are saved to `metrics/parity.csv`.

### 4. Interactive Demo (`app/`)
A Streamlit application (`app/app.py`) provides:
-   **Recommendations**: Generate personal book lists based on user ID and optional genre filters.
-   **Insights**: Interactive EDA visualizations (rating distributions, long-tail analysis, etc.).
-   **Fairness Snapshot**: Visual display of the "gap" in performance metrics across different user groups.

## Quickstart

The project uses a `Makefile` for reproducible workflows.

1.  **Setup Environment**:
    ```bash
    make setup
    ```
    *Creates a virtual environment and installs dependencies from `requirements.txt`.*

2.  **Prepare Data**:
    Ensure `books.csv` and `ratings.csv` are in `data/raw/` (or use the collection scripts to generate them).
    ```bash
    make data
    ```

3.  **Train & Evaluate**:
    ```bash
    make train    # Train ALS model
    make eval     # Calculate ranking metrics (P@10, R@10, NDCG@10)
    make parity   # Run fairness checks
    ```

4.  **Run Application**:
    ```bash
    make app
    ```
    *Launches the Streamlit dashboard.*

## Repository Structure

```text
.
├── app/                 # Streamlit demo application
├── data/                # Raw (input) and Processed (training) data
├── data_collection_scripts/ # Scripts for scraping Goodreads
├── deliverables/        # Project reports and presentations
├── figs/                # Generated figures for EDA and reports
├── metrics/             # Evaluation results (metrics.json, parity.csv)
├── models/              # Serialized model artifacts (ALS factors)
├── scripts/             # Data collection utilities
├── src/                 # Core source code (training, eval, parity)
├── Makefile             # Command-line entry points
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Metrics Snapshot

After running `make eval`, the current model performance is saved to `metrics/metrics.json`:
-   **p10**: Precision at 10
-   **r10**: Recall at 10
-   **ndcg10**: Normalized Discounted Cumulative Gain at 10

## License

This project is licensed under the terms described in the `LICENSE` file.