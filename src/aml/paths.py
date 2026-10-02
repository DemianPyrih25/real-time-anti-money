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
