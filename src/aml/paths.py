"""Volume layout (docs/modal/getting-started.md §3). Modal jobs pass the root /data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path
    dataset: str = "hi_small"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def parquet_dir(self) -> Path:
        return self.root / "parquet" / self.dataset

    @property
    def transactions(self) -> Path:
        return self.parquet_dir / "transactions.parquet"

    @property
    def accounts(self) -> Path:
        return self.parquet_dir / "accounts.parquet"

    @property
    def fx_rates(self) -> Path:
        return self.parquet_dir / "fx_rates.json"

    @property
    def labels(self) -> Path:
        return self.root / "labels" / self.dataset / "labels.parquet"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def mlflow(self) -> Path:
        return self.root / "mlflow"

    def model_dir(self, kind: str, run_key: str) -> Path:
        """e.g. model_dir("rules", key) -> <root>/models/rules/<key>/"""
        return self.root / "models" / kind / run_key

    @property
    def features_root(self) -> Path:
        return self.root / "features" / self.dataset

    def features_dir(self, run_key: str) -> Path:
        """The feature engine's replay outputs: <root>/features/<dataset>/<features_key>/"""
        return self.features_root / run_key

    @property
    def serving_dir(self) -> Path:
        """The serving bundle (`make export`, pulled by `make pull`)."""
        return self.root / "models" / "serving"

    # --- M3 (GNN) ---------------------------------------------------------------------------

    @property
    def optuna_dir(self) -> Path:
        """Optuna trial logs of the GNN HPO: <root>/optuna/<gnn_hpo_key>.jsonl (PLAN path)."""
        return self.root / "optuna"

    def gnn_optuna_log(self, hpo_key: str) -> Path:
        """The GNN HPO trial log, the study's source of truth."""
        return self.optuna_dir / f"{hpo_key}.jsonl"

    def gnn_run_dir(self, run_key: str) -> Path:
        """One (protocol, seed) GNN training run: <root>/models/gnn/<gnn_run_key>/"""
        return self.model_dir("gnn", run_key)

    def gnn_set_dir(self, model: str, run_key: str) -> Path:
        """A GNN stage directory: gnn_bench, gnn_hpo or a set (gnn_causal, gnn_lookahead,
        gnn_lookahead_d10, gnn_pna, gnn_faithful, gnn_dev) -> <root>/models/<model>/<key>/"""
        from aml.models.gnn import DIR_KINDS

        if model not in DIR_KINDS or model == "gnn":
            raise ValueError(f"unknown GNN stage directory kind {model!r}; expected {DIR_KINDS}")
        return self.model_dir(model, run_key)
