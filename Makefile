# Thin aliases for the reproduction commands in PLAN.md §11. Every heavy stage runs on Modal;
# the raw `uv run modal ...` commands work without make (e.g. on Windows).
# After work: `make apps` and stop anything still running (COST_NOTES.md).

export PYTHONIOENCODING := utf-8
# Git Bash / MSYS rewrites arguments that start with "/" (Volume paths) into Windows paths.
export MSYS_NO_PATHCONV := 1

GPU ?= L4
# make gnn-lookahead SEEDS=0 trains look-ahead seed 0 only (then SEEDS=1,2 or all).
SEEDS ?=
MODAL := uv run modal
VOLUME := aml-data

.PHONY: help sync lint fmt test smoke data rules lgbm eval mlflow-ui pull-reports apps clean-local
.PHONY: features-bench features features-verify lgbm-graph export pull
.PHONY: gnn-bench gnn-dev gnn-hpo gnn gnn-lookahead gnn-faithful gnn-pna gnn-plan eval-gnn
.PHONY: demo demo-down bundle-fixture
.PHONY: cases-val cases

help:
	@echo "sync lint fmt test | smoke [GPU=T4] data rules lgbm eval | mlflow-ui pull-reports apps clean-local"
	@echo "M2: features-bench features features-verify rules lgbm-graph eval export pull"
	@echo "M3: smoke GPU=T4 gnn-bench gnn-plan gnn-dev gnn-hpo gnn gnn-lookahead [SEEDS=0] gnn-faithful gnn-pna eval-gnn"
	@echo "M5: pull demo demo-down bundle-fixture (Docker; README > Streaming demo)"
	@echo "M6: cases-val (fit the typology tree, tune the list) cases (test: case packs + full-period parity) pull-reports"

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

# M3: the GNN (M3 spec §12.3, §15.4). Bench first; every GPU job's entrypoint runs the cost gate.
gnn-bench:
	$(MODAL) run --detach -m modal_jobs.gnn_bench

gnn-plan:
	$(MODAL) run -m modal_jobs.train_gnn --plan-only

gnn-dev:
	$(MODAL) run --detach -m modal_jobs.train_gnn --protocol causal --seeds 0 --max-epochs 2 --dev

gnn-hpo:
	$(MODAL) run --detach -m modal_jobs.hpo_gnn

gnn:
	$(MODAL) run --detach -m modal_jobs.train_gnn --protocol causal --final

gnn-lookahead:
	$(MODAL) run --detach -m modal_jobs.train_gnn --protocol lookahead --final$(if $(SEEDS), --seeds $(SEEDS))

gnn-faithful:
	$(MODAL) run --detach -m modal_jobs.train_gnn --protocol faithful --final

gnn-pna:
	$(MODAL) run --detach -m modal_jobs.train_gnn --protocol pna --final

eval-gnn:
	$(MODAL) run -m modal_jobs.evaluate --with-gnn

# M6: case packs over whole periods. Val first: copy reports/typology_tree_val.json to
# configs/typology_tree.json and the tuned thresholds into configs/explain.yaml, then the test
# run measures both (tree primary, decision list baseline) and checks full-period parity.
cases-val:
	$(MODAL) run --detach -m modal_jobs.case_eval --period val

cases:
	$(MODAL) run --detach -m modal_jobs.case_eval --period test

mlflow-ui:
	$(MODAL) serve -m modal_jobs.mlflow_ui

pull-reports:
	$(MODAL) volume get --force $(VOLUME) /reports .

# The serving bundle -> ./serving/ (gitignored).
pull:
	$(MODAL) volume get --force $(VOLUME) /models/serving .

# M5: the streaming demo (CPU-only Docker Compose; needs ./serving from `make pull`).
# Every `up` starts fresh: init recreates the topics and clears the runtime volume.
demo:
	docker compose up --build

demo-down:
	docker compose down --volumes

# A synthetic, Modal-free bundle in ./serving (it refuses to overwrite a pulled real bundle).
bundle-fixture:
	docker compose run --rm --build --no-deps -v ./tests:/app/tests:ro -v ./serving:/out -e PYTHONPATH=/app/src:/app scorer python -m tests.fixtures.serving_bundle --out /out

apps:
	$(MODAL) app list

clean-local:
	rm -rf .pytest_cache .ruff_cache .hypothesis synthetic_out mlruns mlartifacts
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
