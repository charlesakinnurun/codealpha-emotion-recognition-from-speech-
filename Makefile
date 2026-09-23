.PHONY: install dev-install prepare eda baseline train evaluate predict api app test lint lint-fix clean

PYTHON ?= .venv/Scripts/python.exe

## Setup
install: ## Create venv and install runtime deps
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

dev-install: ## Install runtime + dev deps (editable package)
	$(PYTHON) -m pip install -e ".[dev]" -r requirements.txt

## Data pipeline
prepare: ## Build validated metadata + speaker-disjoint splits
	$(PYTHON) scripts/prepare_data.py --config configs/data.yaml

eda: ## Regenerate EDA figures into reports/figures
	$(PYTHON) scripts/eda_figures.py --metadata data/processed/metadata.csv

## Training / evaluation
baseline: ## Classical baselines: CV ablations + one-shot val eval
	$(PYTHON) scripts/run_baselines.py --config configs/baseline.yaml

train:
	$(PYTHON) scripts/train.py $(ARGS)

evaluate:
	$(PYTHON) scripts/evaluate.py $(ARGS)

predict:
	$(PYTHON) scripts/predict.py $(ARGS)

## API
api:
	$(PYTHON) -m uvicorn api.main:app --reload

## Demo
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

## Quality
test: ## Run the full test suite (no dataset download required)
	$(PYTHON) -m pytest

lint: ## Ruff lint + format check
	$(PYTHON) -m ruff check src tests scripts $(wildcard api)
	$(PYTHON) -m ruff format --check src tests scripts $(wildcard api)

lint-fix: ## Auto-fix lint + format issues
	$(PYTHON) -m ruff check --fix src tests scripts $(wildcard api)
	$(PYTHON) -m ruff format src tests scripts $(wildcard api)

clean: ## Remove generated artifacts
	rm -rf .pytest_cache .ruff_cache
	rm -rf reports/experiments/*