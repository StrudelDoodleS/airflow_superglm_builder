"""Seed demo model/package history for inspecting the pricing schema."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.models.config import ModelBuildConfig  # noqa: E402
from pricing_pipeline.publishing.deployment import deploy_rate_package  # noqa: E402
from pricing_pipeline.publishing.lineage import record_model_run  # noqa: E402
from pricing_pipeline.publishing.model_registry import ensure_pricing_model  # noqa: E402
from scripts.load_staging_to_rating_package import publish_rating_package  # noqa: E402
from scripts.load_superglm_excel_to_staging import insert_staging_frames  # noqa: E402
from scripts.pricing_db import get_engine  # noqa: E402


@dataclass(frozen=True)
class DemoLevel:
    level_code: str
    multiplier: float
    exposure_weight: float
    record_count: int
    lower_bound: float | None = None
    upper_bound: float | None = None
    representative_value: float | None = None


@dataclass(frozen=True)
class DemoTerm:
    term_name: str
    feature_name: str
    feature_value_type: str
    level_set_type: str
    term_type: str
    levels: tuple[DemoLevel, ...]


@dataclass(frozen=True)
class DemoPackage:
    model_name: str
    model_label: str
    model_version: str
    target_name: str
    model_type: str
    data_slice_name: str
    base_rate: float
    effective_from: str
    pointer_name: str
    package_status: str
    terms: tuple[DemoTerm, ...]

    @property
    def export_id(self) -> str:
        safe_model = self.model_name.lower()
        safe_version = self.model_version.lower()
        return f"demo__{safe_model}__{safe_version}"


def severity_terms(*, include_brand: bool, brand_b12_multiplier: float) -> tuple[DemoTerm, ...]:
    terms = [
        DemoTerm(
            term_name="VehPower",
            feature_name="VehPower",
            feature_value_type="NUMERIC",
            level_set_type="NUMERIC_BAND",
            term_type="NUMERIC_BANDED_1D",
            levels=(
                DemoLevel("[0, 6)", 0.92, 12_500.0, 12_500, 0.0, 6.0, 3.0),
                DemoLevel("[6, 9)", 1.00, 21_000.0, 21_000, 6.0, 9.0, 7.5),
                DemoLevel("[9, 99)", 1.18, 8_750.0, 8_750, 9.0, 99.0, 54.0),
            ),
        ),
        DemoTerm(
            term_name="Region",
            feature_name="Region",
            feature_value_type="CATEGORICAL",
            level_set_type="CATEGORICAL",
            term_type="CATEGORICAL_MAIN",
            levels=(
                DemoLevel("R11", 0.96, 16_500.0, 16_500),
                DemoLevel("R24", 1.00, 18_200.0, 18_200),
                DemoLevel("R82", 1.12, 7_400.0, 7_400),
            ),
        ),
    ]
    if include_brand:
        terms.append(
            DemoTerm(
                term_name="VehBrand",
                feature_name="VehBrand",
                feature_value_type="CATEGORICAL",
                level_set_type="CATEGORICAL",
                term_type="CATEGORICAL_MAIN",
                levels=(
                    DemoLevel("B1", 0.97, 13_500.0, 13_500),
                    DemoLevel("B6", 1.00, 19_700.0, 19_700),
                    DemoLevel("B12", brand_b12_multiplier, 4_900.0, 4_900),
                ),
            )
        )
    return tuple(terms)


def frequency_terms(
    *,
    expanded_data: bool,
    vehage_uplift: float = 1.0,
) -> tuple[DemoTerm, ...]:
    exposure_scale = 1.65 if expanded_data else 1.0
    count_scale = 1.65 if expanded_data else 1.0

    def scaled_level(
        level_code: str,
        multiplier: float,
        exposure_weight: float,
        record_count: int,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        representative_value: float | None = None,
    ) -> DemoLevel:
        return DemoLevel(
            level_code,
            multiplier,
            exposure_weight * exposure_scale,
            round(record_count * count_scale),
            lower_bound,
            upper_bound,
            representative_value,
        )

    return (
        DemoTerm(
            term_name="VehAge",
            feature_name="VehAge",
            feature_value_type="NUMERIC",
            level_set_type="SPLINE_GRID_1D",
            term_type="DISCRETIZED_SPLINE_1D",
            levels=(
                scaled_level("[0, 1)", 1.34, 8_200.0, 8_200, 0.0, 1.0, 0.5),
                scaled_level("[1, 5)", 1.08, 35_000.0, 35_000, 1.0, 5.0, 3.0),
                scaled_level("[5, 10)", 1.00, 51_000.0, 51_000, 5.0, 10.0, 7.5),
                scaled_level(
                    "[10, 20)",
                    0.88 * vehage_uplift,
                    42_000.0,
                    42_000,
                    10.0,
                    20.0,
                    15.0,
                ),
                scaled_level("[20, 99)", 1.12, 9_800.0, 9_800, 20.0, 99.0, 59.5),
            ),
        ),
        DemoTerm(
            term_name="BonusMalus",
            feature_name="BonusMalus",
            feature_value_type="NUMERIC",
            level_set_type="NUMERIC_BAND",
            term_type="NUMERIC_BANDED_1D",
            levels=(
                scaled_level("[50, 75)", 0.91, 38_000.0, 38_000, 50.0, 75.0, 62.5),
                scaled_level("[75, 100)", 1.00, 74_000.0, 74_000, 75.0, 100.0, 87.5),
                scaled_level("[100, 230)", 1.47, 34_000.0, 34_000, 100.0, 230.0, 165.0),
            ),
        ),
    )


def urban_terms(*, include_density: bool) -> tuple[DemoTerm, ...]:
    terms = [
        DemoTerm(
            term_name="Area",
            feature_name="Area",
            feature_value_type="CATEGORICAL",
            level_set_type="CATEGORICAL",
            term_type="CATEGORICAL_MAIN",
            levels=(
                DemoLevel("A", 0.88, 22_000.0, 22_000),
                DemoLevel("C", 1.00, 27_500.0, 27_500),
                DemoLevel("F", 1.22, 11_200.0, 11_200),
            ),
        ),
        DemoTerm(
            term_name="VehGas",
            feature_name="VehGas",
            feature_value_type="CATEGORICAL",
            level_set_type="CATEGORICAL",
            term_type="CATEGORICAL_MAIN",
            levels=(
                DemoLevel("Regular", 1.00, 38_000.0, 38_000),
                DemoLevel("Diesel", 1.05, 22_700.0, 22_700),
            ),
        ),
    ]
    if include_density:
        terms.append(
            DemoTerm(
                term_name="Density",
                feature_name="Density",
                feature_value_type="NUMERIC",
                level_set_type="NUMERIC_BAND",
                term_type="NUMERIC_BANDED_1D",
                levels=(
                    DemoLevel("[0, 250)", 0.93, 18_100.0, 18_100, 0.0, 250.0, 125.0),
                    DemoLevel("[250, 1000)", 1.00, 24_600.0, 24_600, 250.0, 1000.0, 625.0),
                    DemoLevel("[1000, 99999)", 1.16, 18_000.0, 18_000, 1000.0, 99999.0, 50500.0),
                ),
            )
        )
    return tuple(terms)


def demo_packages() -> list[DemoPackage]:
    return [
        DemoPackage(
            model_name="MTPL_FREQ_DEMO",
            model_label="Demo frequency model on freMTPL policy sample",
            model_version="20260429_v1_base",
            target_name="ClaimNb",
            model_type="superglm_poisson_demo",
            data_slice_name="frequency_policy_sample",
            base_rate=0.064,
            effective_from="2026-04-29",
            pointer_name="MTPL_FREQ_DEMO_UAT",
            package_status="PUBLISHED",
            terms=frequency_terms(expanded_data=False),
        ),
        DemoPackage(
            model_name="MTPL_FREQ_DEMO",
            model_label="Demo frequency model on expanded freMTPL policy sample",
            model_version="20260429_v2_more_data",
            target_name="ClaimNb",
            model_type="superglm_poisson_demo",
            data_slice_name="frequency_expanded_policy_sample",
            base_rate=0.061,
            effective_from="2026-04-29",
            pointer_name="MTPL_FREQ_DEMO_UAT",
            package_status="PUBLISHED",
            terms=frequency_terms(expanded_data=True),
        ),
        DemoPackage(
            model_name="MTPL_FREQ_DEMO",
            model_label="Demo frequency model with manual VehAge relativity uplift",
            model_version="20260429_v3_manual_vehage_uplift",
            target_name="ClaimNb",
            model_type="superglm_poisson_demo",
            data_slice_name="frequency_expanded_policy_sample",
            base_rate=0.061,
            effective_from="2026-04-29",
            pointer_name="MTPL_FREQ_DEMO_UAT",
            package_status="PUBLISHED",
            terms=frequency_terms(expanded_data=True, vehage_uplift=1.10),
        ),
        DemoPackage(
            model_name="MTPL_SEV_DEMO",
            model_label="Demo severity model on large-loss freMTPL slice",
            model_version="20260429_v1_base",
            target_name="ClaimAmount",
            model_type="superglm_gamma_demo",
            data_slice_name="severity_large_loss_slice",
            base_rate=1450.0,
            effective_from="2026-04-29",
            pointer_name="MTPL_SEV_DEMO_UAT",
            package_status="PUBLISHED",
            terms=severity_terms(include_brand=True, brand_b12_multiplier=1.20),
        ),
    ]


def package_feature_names(package: DemoPackage) -> set[str]:
    return {term.feature_name for term in package.terms}


def package_level_multiplier(
    package: DemoPackage,
    *,
    term_name: str,
    level_code: str,
) -> float:
    for term in package.terms:
        if term.term_name != term_name:
            continue
        for level in term.levels:
            if level.level_code == level_code:
                return level.multiplier
    raise KeyError(f"{package.model_name} {package.model_version} {term_name}={level_code}")


def manifest_id_for_package(package: DemoPackage) -> str:
    return f"demo__{package.data_slice_name}__{package.effective_from.replace('-', '')}"


def build_staging_frames(
    package: DemoPackage,
    *,
    created_by: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    export_df = pd.DataFrame(
        [
            {
                "export_id": package.export_id,
                "model_name": package.model_name,
                "model_version": package.model_version,
                "base_rate": package.base_rate,
                "effective_from_date": package.effective_from,
                "effective_to_date": None,
                "source_file": f"demo://{package.data_slice_name}",
                "created_by": created_by,
            }
        ]
    )

    rate_rows = []
    level_rows = []
    row_id = 0
    for sequence_no, term in enumerate(package.terms, start=1):
        for order_index, level in enumerate(term.levels, start=1):
            row_id += 1
            cell_key = f"{term.term_name}={level.level_code}"
            rate_rows.append(
                {
                    "export_id": package.export_id,
                    "row_id": row_id,
                    "term_name": term.term_name,
                    "term_type": term.term_type,
                    "sequence_no": sequence_no,
                    "cell_key_text": cell_key,
                    "multiplier": level.multiplier,
                    "log_coefficient": math.log(level.multiplier),
                    "exposure_weight": level.exposure_weight,
                    "record_count": level.record_count,
                    "is_reference": int(math.isclose(level.multiplier, 1.0)),
                    "is_default": 0,
                }
            )
            level_rows.append(
                {
                    "export_id": package.export_id,
                    "row_id": row_id,
                    "position_no": 1,
                    "feature_name": term.feature_name,
                    "feature_value_type": term.feature_value_type,
                    "level_set_name": f"{term.feature_name}__{package.export_id}",
                    "level_set_type": term.level_set_type,
                    "level_code": level.level_code,
                    "level_label": level.level_code,
                    "order_index": order_index,
                    "lower_bound": level.lower_bound,
                    "upper_bound": level.upper_bound,
                    "representative_value": level.representative_value,
                    "is_missing": 0,
                    "is_other": 0,
                }
            )

    return export_df, pd.DataFrame(rate_rows), pd.DataFrame(level_rows)


def ensure_demo_manifest(engine, package: DemoPackage, *, created_by: str) -> str:
    manifest_id = manifest_id_for_package(package)
    with engine.begin() as con:
        con.execute(
            text(
                """
                MERGE pricing.DATASET_MANIFEST WITH (HOLDLOCK) AS tgt
                USING (
                    SELECT
                        :manifest_id AS manifest_id,
                        :dataset_name AS dataset_name,
                        :source_system AS source_system,
                        :data_as_of_date AS data_as_of_date,
                        :row_count AS row_count,
                        :target_column AS target_column,
                        :created_by AS created_by
                ) AS src
                ON tgt.manifest_id = src.manifest_id
                WHEN MATCHED THEN
                    UPDATE SET
                        dataset_name = src.dataset_name,
                        source_system = src.source_system,
                        data_as_of_date = src.data_as_of_date,
                        row_count = src.row_count,
                        target_column = src.target_column
                WHEN NOT MATCHED THEN
                    INSERT (
                        manifest_id,
                        dataset_name,
                        source_system,
                        data_as_of_date,
                        row_count,
                        pk_columns_json,
                        target_column,
                        weight_column,
                        created_by
                    )
                    VALUES (
                        src.manifest_id,
                        src.dataset_name,
                        src.source_system,
                        src.data_as_of_date,
                        src.row_count,
                        '["demo_row_id"]',
                        src.target_column,
                        'Exposure',
                        src.created_by
                    );
                """
            ),
            {
                "manifest_id": manifest_id,
                "dataset_name": package.data_slice_name,
                "source_system": "demo_seed",
                "data_as_of_date": package.effective_from,
                "row_count": sum(
                    level.record_count for term in package.terms for level in term.levels
                ),
                "target_column": package.target_name,
                "created_by": created_by,
            },
        )
    return manifest_id


def seed_package(engine, package: DemoPackage, *, created_by: str) -> int:
    model_id = ensure_pricing_model(
        engine,
        model_name=package.model_name,
        model_label=package.model_label,
        target_name=package.target_name,
        model_type=package.model_type,
        created_by=created_by,
    )
    export_df, rate_df, level_df = build_staging_frames(package, created_by=created_by)
    args = argparse.Namespace(
        export_id=package.export_id,
        model_name=package.model_name,
        model_label=package.model_label,
        target_name=package.target_name,
        model_type=package.model_type,
        model_status="ACTIVE",
        created_by=created_by,
        replace=True,
        model_id=model_id,
    )
    insert_staging_frames(engine, args, export_df, rate_df, level_df)
    rate_package_id = publish_rating_package(
        engine,
        export_id=package.export_id,
        created_by=created_by,
        package_status=package.package_status,
    )
    manifest_id = ensure_demo_manifest(engine, package, created_by=created_by)
    if package.package_status == "PUBLISHED":
        with engine.begin() as con:
            expected_current_rate_package_id = con.execute(
                text("""
                    SELECT rate_package_id
                    FROM pricing.PRICING_MODEL_DEPLOYMENT
                    WHERE model_id = :model_id
                      AND deployment_slot = :deployment_slot
                      AND effective_to_ts IS NULL
                """),
                {
                    "model_id": model_id,
                    "deployment_slot": package.pointer_name,
                },
            ).scalar_one_or_none()
        deploy_rate_package(
            engine,
            ModelBuildConfig(
                model_name=package.model_name,
                model_label=package.model_label,
                target_name=package.target_name,
                model_type=package.model_type,
                deployment_slot=package.pointer_name,
                default_package_status="PUBLISHED",
            ),
            rate_package_id=rate_package_id,
            expected_current_rate_package_id=expected_current_rate_package_id,
            deployment_slot=package.pointer_name,
            deployment_reason="seed demo package",
            deployed_by=created_by,
            model_id=model_id,
        )
    record_model_run(
        engine,
        dag_id="demo_model_seed",
        airflow_run_id=package.export_id,
        mlflow_run_id=f"demo-{package.export_id}",
        manifest_id=manifest_id,
        export_id=package.export_id,
        model_id=model_id,
        model_name=package.model_name,
        model_version=package.model_version,
        rate_package_id=rate_package_id,
        rating_workbook_path=f"demo://{package.export_id}/rating_tables.xlsx",
        run_status="SUCCESS",
        created_by=created_by,
    )
    return rate_package_id


def seed_demo_packages(*, created_by: str = "demo_seed") -> list[tuple[DemoPackage, int]]:
    engine = get_engine()
    seeded = []
    for package in demo_packages():
        rate_package_id = seed_package(engine, package, created_by=created_by)
        seeded.append((package, rate_package_id))
    return seeded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--created-by", default="demo_seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for package, rate_package_id in seed_demo_packages(created_by=args.created_by):
        print(
            "seeded "
            f"model={package.model_name} "
            f"version={package.model_version} "
            f"rate_package_id={rate_package_id} "
            f"slot={package.pointer_name}"
        )


if __name__ == "__main__":
    main()
