PY=python
SEED?=42

.PHONY: setup data train eval parity app

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	$(PY) -m src.data_prep --seed $(SEED)

train:
	$(PY) -m src.als_train --seed $(SEED)

eval:
	$(PY) -m src.eval_ranking --seed $(SEED)

parity:
	$(PY) -m src.parity --seed $(SEED)

app:
	streamlit run app/app.py

# Data collection
crawl_books:
	python -m scripts.get_rating_of_books --seeds src/ingest/assets/seed_books.csv --out data/raw/reviews --resume

crawl_users:
	python -m scripts.get_rating_of_users --in data/raw/reviews --out data/raw/users --resume

count_books:
	python -m scripts.count_books --in data/raw/users --out data/interim/book_counts.csv

build_dataset:
	python -m scripts.build_goodreads_dataset --in data/raw/users --out data/processed/goodreads_dataset.csv
