# Thin aliases for the reproduction commands in PLAN.md §11. Every heavy stage runs on Modal;
# the raw `uv run modal ...` commands work without make (e.g. on Windows).
# After work: `make apps` and stop anything still running (COST_NOTES.md).

export PYTHONIOENCODING := utf-8
# Git Bash / MSYS rewrites arguments that start with "/" (Volume paths) into Windows paths.
export MSYS_NO_PATHCONV := 1

GPU ?= L4
MODAL := uv run modal
VOLUME := aml-data

.PHONY: help sync lint fmt test smoke data rules lgbm eval mlflow-ui pull-reports apps clean-local
.PHONY: features-bench features features-verify lgbm-graph export pull

help:
	@echo "sync lint fmt test | smoke [GPU=T4] data rules lgbm eval | mlflow-ui pull-reports apps clean-local"
	@echo "M2: features-bench features features-verify rules lgbm-graph eval export pull"

sync:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

smoke:
	$(MODAL) run -m modal_jobs.smoke --gpu $(GPU)

data:
	$(MODAL) run -m modal_jobs.prepare_data

rules:
	$(MODAL) run -m modal_jobs.rules

lgbm:
	$(MODAL) run --detach -m modal_jobs.train_lgbm

eval:
	$(MODAL) run -m modal_jobs.evaluate

# M2: the feature-engine replay (bench first: it sizes memory and the full run's timeout).
features-bench:
	$(MODAL) run --detach -m modal_jobs.build_features --mode bench

features:
	$(MODAL) run --detach -m modal_jobs.build_features --mode full

features-verify:
	$(MODAL) run --detach -m modal_jobs.build_features --mode verify

lgbm-graph:
	$(MODAL) run --detach -m modal_jobs.train_lgbm --feature-set graph

export:
	$(MODAL) run -m modal_jobs.export

mlflow-ui:
	$(MODAL) serve -m modal_jobs.mlflow_ui

pull-reports:
	$(MODAL) volume get --force $(VOLUME) /reports .

# The serving bundle -> ./serving/ (gitignored).
pull:
	$(MODAL) volume get --force $(VOLUME) /models/serving .

apps:
	$(MODAL) app list

clean-local:
	rm -rf .pytest_cache .ruff_cache .hypothesis synthetic_out mlruns mlartifacts
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
