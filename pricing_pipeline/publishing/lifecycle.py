from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RatePackageSelector:
    rate_package_id: int | None = None
    package_version: int | None = None

    def __post_init__(self) -> None:
        selected = [
            self.rate_package_id is not None,
            self.package_version is not None,
        ]
        if sum(selected) != 1:
            raise ValueError("exactly one rate package selector is required")


@dataclass(frozen=True)
class PublishResult:
    mlflow_run_id: str
    export_id: str
    rate_package_id: int
    package_version: int
    rating_workbook_path: str
    package_status: str = "PUBLISHED"
    was_existing: bool = False


@dataclass(frozen=True)
class DeploymentResult:
    model_id: int
    deployment_slot: str
    previous_rate_package_id: int | None
    rate_package_id: int
    package_version: int
    deployed_by: str
    deployment_reason: str


@dataclass(frozen=True)
class RatePackageSnapshot:
    metadata: dict[str, Any]
    terms: pd.DataFrame
    rate_cells: pd.DataFrame
    cell_levels: pd.DataFrame
    compiled_rate_cells: pd.DataFrame
    compiled_1d_bands: pd.DataFrame


@dataclass(frozen=True)
class RatePackageRevisionResult:
    rate_package_id: int
    package_version: int
    parent_rate_package_id: int
    changed_rate_cell_count: int
    base_rate_changed: bool
    diff_summary: pd.DataFrame


@dataclass(frozen=True)
class PredictionComparison:
    summary: dict[str, float]
    changed_rows: pd.DataFrame
