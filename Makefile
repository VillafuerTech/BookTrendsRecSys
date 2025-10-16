PY=python
SEED?=42

.PHONY: setup data train eval parity app report

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

report:
	$(MAKE) -C deliverables/2025-10-16-spotlight/report report-all
