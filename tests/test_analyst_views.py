from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)


_ANALYST_VIEWS = (
    "V_FINAL_MODEL_RELATIVITY",
    "V_MODEL_VALIDATION_SPLIT",
    "V_MODEL_VALIDATION_SUMMARY",
    "V_MODEL_VALIDATION_SPLIT_RELATIVITY",
    "V_CURRENT_DATASET_VALIDATION_SPLIT",
)


def _seed_split_geometry(connection: sqlite3.Connection) -> None:
    manifests = (
        (
            "manifest-old-kfold",
            "motor_portfolio",
            100,
            "2025-01-01T00:00:00Z",
        ),
        ("manifest-kfold", "motor_portfolio", 100, "2026-01-01T00:00:00Z"),
        ("manifest-column-kfold", "column_portfolio", 90, "2026-01-02T00:00:00Z"),
        ("manifest-train-test", "severity_portfolio", 100, "2026-01-03T00:00:00Z"),
        ("manifest-column-holdout", "holdout_portfolio", 80, "2026-01-04T00:00:00Z"),
        ("manifest-custom", "custom_portfolio", 60, "2026-01-05T00:00:00Z"),
    )
    connection.executemany(
        """
        INSERT INTO DATASET_MANIFEST (
            manifest_id, dataset_name, source_system, data_as_of_date,
            row_count, pk_columns_json, target_column, weight_column,
            model_frame_sha256, frame_hash_metadata_json, data_as_of_column,
            created_ts, created_by
        ) VALUES (?, ?, 'semantic-fixture', '2025-12-31', ?, '["policy_id"]',
                  'ClaimNb', 'Exposure', ?, '{"algorithm":"sha256"}',
                  'snapshot_date', ?, 'pytest')
        """,
        [
            (manifest_id, dataset_name, row_count, str(index) * 64, created_ts)
            for index, (manifest_id, dataset_name, row_count, created_ts) in enumerate(
                manifests,
                start=1,
            )
        ],
    )

    split_sets = (
        (
            "split-old-kfold",
            "manifest-old-kfold",
            "sklearn.model_selection.KFold",
            {"method": "kfold", "n_splits": 1},
            100,
            1,
            "2025-01-01T00:00:00Z",
        ),
        (
            "split-kfold",
            "manifest-kfold",
            "sklearn.model_selection.KFold",
            {"method": "kfold", "n_splits": 2},
            100,
            2,
            "2026-01-01T00:00:00Z",
        ),
        (
            "split-column-kfold",
            "manifest-column-kfold",
            "source_column",
            {"method": "column_kfold", "column": "fold_number"},
            90,
            3,
            "2026-01-02T00:00:00Z",
        ),
        (
            "split-train-test",
            "manifest-train-test",
            "sklearn.model_selection.train_test_split",
            {"method": "train_test_split", "test_size": 0.25},
            100,
            1,
            "2026-01-03T00:00:00Z",
        ),
        (
            "split-column-holdout",
            "manifest-column-holdout",
            "source_column",
            {"method": "column_holdout", "column": "is_validation"},
            80,
            1,
            "2026-01-04T00:00:00Z",
        ),
        (
            "split-custom",
            "manifest-custom",
            "custom",
            {"method": "custom"},
            60,
            2,
            "2026-01-05T00:00:00Z",
        ),
    )
    connection.executemany(
        """
        INSERT INTO CV_SPLIT_SET (
            split_set_id, manifest_id, split_mode, splitter_class,
            splitter_params_json, row_order_sha256, row_count, fold_count,
            created_ts, created_by
        ) VALUES (?, ?, 'REPLAYABLE', ?, ?, ?, ?, ?, ?, 'pytest')
        """,
        [
            (
                split_set_id,
                manifest_id,
                splitter_class,
                json.dumps(params, sort_keys=True),
                str(index) * 64,
                row_count,
                fold_count,
                created_ts,
            )
            for index, (
                split_set_id,
                manifest_id,
                splitter_class,
                params,
                row_count,
                fold_count,
                created_ts,
            ) in enumerate(split_sets, start=1)
        ],
    )
    connection.executemany(
        "INSERT INTO CV_FOLD (split_set_id, fold_no, n_train, n_test) VALUES (?, ?, ?, ?)",
        (
            ("split-old-kfold", 1, 50, 50),
            ("split-kfold", 1, 50, 50),
            ("split-kfold", 2, 50, 50),
            ("split-column-kfold", 1, 60, 30),
            ("split-column-kfold", 2, 60, 30),
            ("split-column-kfold", 3, 60, 30),
            ("split-train-test", 1, 75, 25),
            ("split-column-holdout", 1, 70, 10),
            ("split-custom", 1, 40, 20),
            ("split-custom", 2, 40, 20),
        ),
    )


def _seed_packages_and_runs(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO PRICING_MODEL (
            model_id, model_name, model_label, target_name, model_type,
            created_by
        ) VALUES (1, 'frequency_model', 'Frequency model', 'ClaimNb', 'SuperGLM', 'pytest')
        """
    )

    log_metadata = json.dumps(
        {"model": {"family": "Poisson", "family_params": {"dispersion": 1.0}, "link": "log"}},
        sort_keys=True,
    )
    identity_metadata = json.dumps(
        {"model": {"family": "Gaussian", "family_params": {}, "link": "identity"}},
        sort_keys=True,
    )
    packages = (
        (10, None, "v1", 1, "manifest-kfold", "split-kfold", log_metadata, "a" * 64),
        (11, 10, "v1", 2, "manifest-kfold", "split-kfold", log_metadata, None),
        (
            20,
            None,
            "v2",
            3,
            "manifest-column-kfold",
            "split-column-kfold",
            log_metadata,
            "b" * 64,
        ),
        (
            30,
            None,
            "v3",
            4,
            "manifest-train-test",
            "split-train-test",
            identity_metadata,
            "c" * 64,
        ),
        (
            40,
            None,
            "v4",
            5,
            "manifest-column-holdout",
            "split-column-holdout",
            log_metadata,
            "d" * 64,
        ),
        (50, None, "v5", 6, "manifest-custom", "split-custom", log_metadata, "e" * 64),
    )
    connection.executemany(
        """
        INSERT INTO PRICING_RATE_PACKAGE (
            rate_package_id, parent_rate_package_id, model_id, model_name,
            model_version, package_version, base_rate, package_status,
            source_export_id, package_metadata_json, revision_metadata_json,
            build_fingerprint_sha256, manifest_id, split_set_id, created_by
        ) VALUES (?, ?, 1, 'frequency_model', ?, ?, 0.05, 'PUBLISHED', ?, ?, ?, ?, ?, ?, 'pytest')
        """,
        [
            (
                package_id,
                parent_id,
                model_version,
                package_version,
                f"export-{package_id}",
                metadata,
                json.dumps(
                    (
                        {"kind": "ORIGINAL"}
                        if parent_id is None
                        else {"kind": "EDITED", "reason": "manual underwriting adjustment"}
                    ),
                    sort_keys=True,
                ),
                fingerprint,
                manifest_id,
                split_set_id,
            )
            for (
                package_id,
                parent_id,
                model_version,
                package_version,
                manifest_id,
                split_set_id,
                metadata,
                fingerprint,
            ) in packages
        ],
    )

    runs = (
        ("run-root", None, "v1", 10, "manifest-kfold", "split-kfold", "COMPLETE", None, "run-root"),
        (
            "run-child",
            "run-root",
            "v1",
            11,
            "manifest-kfold",
            "split-kfold",
            None,
            None,
            "run-root",
        ),
        (
            "run-column-kfold",
            None,
            "v2",
            20,
            "manifest-column-kfold",
            "split-column-kfold",
            "UNAVAILABLE",
            "curve comparison not requested",
            "run-column-kfold",
        ),
        (
            "run-train-test",
            None,
            "v3",
            30,
            "manifest-train-test",
            "split-train-test",
            "COMPLETE",
            None,
            "run-train-test",
        ),
        (
            "run-column-holdout",
            None,
            "v4",
            40,
            "manifest-column-holdout",
            "split-column-holdout",
            "UNAVAILABLE",
            "unsupported interaction surface",
            "run-column-holdout",
        ),
        (
            "run-custom",
            None,
            "v5",
            50,
            "manifest-custom",
            "split-custom",
            "UNAVAILABLE",
            "custom scoring omitted curve capture",
            "run-custom",
        ),
    )
    connection.executemany(
        """
        INSERT INTO MODEL_RUN (
            model_run_id, parent_model_run_id, model_id, model_version,
            export_id, manifest_id, split_set_id, rate_package_id, model_name,
            rating_workbook_path, rating_workbook_sha256,
            candidate_python_version, candidate_superglm_version,
            candidate_superglm_git_sha, model_source_sha256,
            builder_source_sha256, materialized_split_sha256, runtime_sha256,
            candidate_superglm_sha256, validation_curve_status,
            validation_curve_reason, validation_source_model_run_id, created_by
        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, 'frequency_model', 'rating.xlsx', ?,
                  '3.14.0', '0.10.0', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pytest')
        """,
        [
            (
                run_id,
                parent_run_id,
                model_version,
                f"export-{package_id}",
                manifest_id,
                split_set_id,
                package_id,
                str(index) * 64,
                "0123456789abcdef0123456789abcdef01234567",
                "1" * 64,
                "2" * 64,
                "3" * 64,
                "4" * 64,
                "5" * 64,
                curve_status,
                curve_reason,
                validation_source,
            )
            for index, (
                run_id,
                parent_run_id,
                model_version,
                package_id,
                manifest_id,
                split_set_id,
                curve_status,
                curve_reason,
                validation_source,
            ) in enumerate(runs, start=1)
        ],
    )


def _seed_metrics_and_curves(connection: sqlite3.Connection) -> None:
    metrics = (
        ("run-root", "split-kfold", 1, "deviance", 1.0),
        ("run-root", "split-kfold", 1, "nll", 2.0),
        ("run-root", "split-kfold", 1, "gini", 0.2),
        ("run-root", "split-kfold", 2, "deviance", 3.0),
        ("run-root", "split-kfold", 2, "nll", 6.0),
        ("run-root", "split-kfold", 2, "gini", 0.6),
        ("run-column-kfold", "split-column-kfold", 1, "deviance", 2.0),
        ("run-column-kfold", "split-column-kfold", 1, "nll", 1.0),
        ("run-column-kfold", "split-column-kfold", 1, "gini", 0.1),
        ("run-column-kfold", "split-column-kfold", 2, "deviance", 4.0),
        ("run-column-kfold", "split-column-kfold", 2, "nll", 3.0),
        ("run-column-kfold", "split-column-kfold", 2, "gini", 0.3),
        ("run-column-kfold", "split-column-kfold", 3, "deviance", 6.0),
        ("run-column-kfold", "split-column-kfold", 3, "nll", 5.0),
        ("run-column-kfold", "split-column-kfold", 3, "gini", 0.5),
        ("run-train-test", "split-train-test", 1, "deviance", 5.0),
        ("run-train-test", "split-train-test", 1, "nll", 7.0),
        ("run-train-test", "split-train-test", 1, "gini", 0.4),
        ("run-column-holdout", "split-column-holdout", 1, "deviance", 8.0),
        ("run-column-holdout", "split-column-holdout", 1, "nll", 9.0),
        ("run-column-holdout", "split-column-holdout", 1, "gini", 0.7),
        ("run-custom", "split-custom", 1, "deviance", 10.0),
        ("run-custom", "split-custom", 2, "deviance", 14.0),
    )
    connection.executemany(
        """
        INSERT INTO CV_FOLD_METRIC (
            model_run_id, split_set_id, fold_no, metric_name, metric_value
        ) VALUES (?, ?, ?, ?, ?)
        """,
        metrics,
    )

    connection.executemany(
        """
        INSERT INTO CV_SPLIT_CURVE_POINT (
            model_run_id, split_set_id, split_no, term_name, point_no,
            point_kind, x_numeric, level_text, eta_contribution, relativity,
            support_value, reference_value, reference_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                "run-root",
                "split-kfold",
                1,
                "Area",
                1,
                "LEVEL",
                None,
                "London",
                0.0,
                1.0,
                50.0,
                None,
                "London",
            ),
            (
                "run-root",
                "split-kfold",
                1,
                "CurveOnly",
                1,
                "NUMERIC",
                1.0,
                None,
                0.2,
                math.exp(0.2),
                40.0,
                1.0,
                None,
            ),
            (
                "run-root",
                "split-kfold",
                2,
                "Area",
                1,
                "LEVEL",
                None,
                "North",
                0.3,
                math.exp(0.3),
                50.0,
                None,
                "London",
            ),
            (
                "run-root",
                "split-kfold",
                2,
                "CurveOnly",
                1,
                "NUMERIC",
                1.0,
                None,
                0.4,
                math.exp(0.4),
                35.0,
                1.0,
                None,
            ),
            (
                "run-train-test",
                "split-train-test",
                1,
                "Severity",
                1,
                "NUMERIC",
                100.0,
                None,
                2.5,
                None,
                25.0,
                100.0,
                None,
            ),
        ),
    )


def _seed_final_rating_tables(connection: sqlite3.Connection) -> None:
    connection.executemany(
        """
        INSERT INTO PRICING_TERM (
            term_id, rate_package_id, term_name, term_type, sequence_no
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            (101, 10, "Area", "CATEGORICAL_MAIN", 1),
            (102, 10, "Density", "NUMERIC_MAIN", 2),
            (103, 10, "Age_Region", "CATEGORICAL_INTERACTION", 3),
            (104, 10, "VehicleAge", "NUMERIC_BANDED_1D", 4),
            (111, 11, "Area", "CATEGORICAL_MAIN", 1),
            (112, 11, "Density", "NUMERIC_MAIN", 2),
            (113, 11, "Age_Region", "CATEGORICAL_INTERACTION", 3),
            (114, 11, "VehicleAge", "NUMERIC_BANDED_1D", 4),
        ),
    )
    connection.execute(
        """
        INSERT INTO PRICING_FEATURE (
            feature_id, feature_name, feature_value_type
        ) VALUES (1, 'VehicleAge', 'NUMERIC')
        """
    )
    connection.execute(
        """
        INSERT INTO PRICING_FEATURE_LEVEL_SET (
            level_set_id, model_id, feature_id, level_set_name, level_set_type
        ) VALUES (1, 1, 1, 'vehicle_age_bands', 'NUMERIC_BAND')
        """
    )
    connection.execute(
        """
        INSERT INTO PRICING_FEATURE_LEVEL (
            feature_level_id, level_set_id, level_code, level_label,
            order_index, lower_bound, upper_bound, representative_value
        ) VALUES (1, 1, 'young', 'Young', 1, 0.0, 5.0, 2.5)
        """
    )
    connection.executemany(
        """
        INSERT INTO PRICING_RATE_CELL (
            cell_id, term_id, cell_key_text, cell_key_digest, multiplier,
            log_coefficient, exposure_weight, record_count,
            is_reference, is_default
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (1004, 104, "VehicleAge=young", "root-band", 1.10, 0.10, 45.0, 45, 1, 0),
            (1104, 114, "VehicleAge=young", "child-band", 1.25, 0.22, 45.0, 45, 1, 0),
        ),
    )
    connection.executemany(
        """
        INSERT INTO PRICING_RATE_CELL_LEVEL (
            cell_id, position_no, feature_level_id
        ) VALUES (?, 1, 1)
        """,
        ((1004,), (1104,)),
    )

    compiled_cells = (
        (
            10,
            101,
            "root-area",
            "Area",
            "CATEGORICAL_MAIN",
            1,
            "Area=London=Area",
            1.10,
            0.10,
            60.0,
            60,
            0,
            1,
        ),
        (
            10,
            102,
            "root-density",
            "Density",
            "NUMERIC_MAIN",
            2,
            "Density=per_unit",
            1.05,
            0.05,
            100.0,
            100,
            0,
            1,
        ),
        (
            10,
            103,
            "root-interaction",
            "Age_Region",
            "CATEGORICAL_INTERACTION",
            3,
            "Age=20|Region=Area=North",
            1.20,
            0.20,
            20.0,
            20,
            0,
            1,
        ),
        (
            10,
            104,
            "root-band",
            "VehicleAge",
            "NUMERIC_BANDED_1D",
            4,
            "VehicleAge=young",
            1.10,
            0.10,
            45.0,
            45,
            0,
            1,
        ),
        (
            11,
            111,
            "child-area",
            "Area",
            "CATEGORICAL_MAIN",
            1,
            "Area=London=Area",
            1.50,
            0.40,
            60.0,
            60,
            0,
            1,
        ),
        (
            11,
            112,
            "child-density",
            "Density",
            "NUMERIC_MAIN",
            2,
            "Density=per_unit",
            1.08,
            0.08,
            100.0,
            100,
            0,
            1,
        ),
        (
            11,
            113,
            "child-interaction",
            "Age_Region",
            "CATEGORICAL_INTERACTION",
            3,
            "Age=20|Region=Area=North",
            1.30,
            0.26,
            20.0,
            20,
            0,
            1,
        ),
        (
            11,
            114,
            "child-band",
            "VehicleAge",
            "NUMERIC_BANDED_1D",
            4,
            "VehicleAge=young",
            1.25,
            0.22,
            45.0,
            45,
            0,
            1,
        ),
    )
    connection.executemany(
        """
        INSERT INTO PRICING_COMPILED_RATE_CELL (
            rate_package_id, term_id, cell_key_digest, term_name, term_type,
            sequence_no, cell_key_text, multiplier, log_coefficient,
            exposure_weight, record_count, is_default, is_reference
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        compiled_cells,
    )
    connection.executemany(
        """
        INSERT INTO PRICING_COMPILED_1D_RATE_BAND (
            rate_package_id, term_id, feature_level_id, term_name, feature_name,
            level_code, sort_order, lower_bound, upper_bound,
            representative_value, multiplier, log_coefficient
        ) VALUES (?, ?, 1, 'VehicleAge', 'VehicleAge', 'young', 1,
                  0.0, 5.0, 2.5, ?, ?)
        """,
        ((10, 104, 1.10, 0.10), (11, 114, 1.25, 0.22)),
    )


@pytest.fixture
def analyst_connection(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    paths = {
        "pricing": tmp_path / "pricing.sqlite",
        "pricing_stg": tmp_path / "pricing_stg.sqlite",
        "mlops": tmp_path / "mlops.sqlite",
    }
    engine = sqlite_engine_with_offline_schemas(paths)
    apply_offline_ddl(engine)
    engine.dispose()

    connection = sqlite3.connect(paths["pricing"])
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    _seed_split_geometry(connection)
    _seed_packages_and_runs(connection)
    _seed_metrics_and_curves(connection)
    _seed_final_rating_tables(connection)
    connection.commit()
    try:
        yield connection
    finally:
        connection.close()


def test_final_view_contains_only_each_packages_actual_final_fit(
    analyst_connection: sqlite3.Connection,
) -> None:
    rows = analyst_connection.execute(
        """
        SELECT *
        FROM V_FINAL_MODEL_RELATIVITY
        ORDER BY rate_package_id, term_sequence_no
        """
    ).fetchall()

    assert len(rows) == 8
    assert {row["rate_package_id"] for row in rows} == {10, 11}
    assert {row["model_fit_scope"] for row in rows} == {"PACKAGE_FINAL_MODEL"}
    assert "feature_name" not in rows[0].keys()

    by_package_and_term = {(row["rate_package_id"], row["term_name"]): row for row in rows}
    root_area = by_package_and_term[(10, "Area")]
    child_area = by_package_and_term[(11, "Area")]
    assert root_area["term_level"] == "London=Area"
    assert root_area["eta_contribution"] == pytest.approx(0.10)
    assert child_area["term_level"] == "London=Area"
    assert child_area["eta_contribution"] == pytest.approx(0.40)
    assert child_area["package_kind"] == "EDITED"
    assert child_area["parent_package_version"] == 1
    assert child_area["edit_reason"] == "manual underwriting adjustment"

    assert by_package_and_term[(10, "Density")]["term_level"] == "per_unit"
    assert by_package_and_term[(10, "Age_Region")]["term_level"] == "Age=20|Region=Area=North"
    assert by_package_and_term[(10, "VehicleAge")]["term_level"] == "young"
    assert by_package_and_term[(10, "VehicleAge")]["relativity_source"] == "1D_RATE_BAND"
    assert "CurveOnly" not in {row["term_name"] for row in rows}


def test_validation_split_view_has_one_held_out_row_per_strategy_split(
    analyst_connection: sqlite3.Connection,
) -> None:
    rows = analyst_connection.execute(
        """
        SELECT *
        FROM V_MODEL_VALIDATION_SPLIT
        ORDER BY rate_package_id, validation_split_no
        """
    ).fetchall()

    expected = {
        10: ("kfold", (1, 2), "DIRECT", "run-root"),
        11: ("kfold", (1, 2), "INHERITED_FROM_PARENT", "run-root"),
        20: ("column_kfold", (1, 2, 3), "DIRECT", "run-column-kfold"),
        30: ("train_test_split", (1,), "DIRECT", "run-train-test"),
        40: ("column_holdout", (1,), "DIRECT", "run-column-holdout"),
        50: ("custom", (1, 2), "DIRECT", "run-custom"),
    }
    for package_id, (method, split_numbers, evidence, source_run_id) in expected.items():
        package_rows = [row for row in rows if row["rate_package_id"] == package_id]
        assert tuple(row["validation_split_no"] for row in package_rows) == split_numbers
        assert {row["split_method"] for row in package_rows} == {method}
        assert {row["validation_evidence"] for row in package_rows} == {evidence}
        assert {row["validation_source_model_run_id"] for row in package_rows} == {source_run_id}

    child_rows = [row for row in rows if row["rate_package_id"] == 11]
    assert [(row["deviance"], row["nll"], row["gini"]) for row in child_rows] == [
        (1.0, 2.0, 0.2),
        (3.0, 6.0, 0.6),
    ]
    assert {row["parent_package_version"] for row in child_rows} == {1}
    assert {row["validation_source_package_version"] for row in child_rows} == {1}

    custom_rows = [row for row in rows if row["rate_package_id"] == 50]
    assert [row["deviance"] for row in custom_rows] == [10.0, 14.0]
    assert all(row["nll"] is None and row["gini"] is None for row in custom_rows)
    grain = {
        (row["rate_package_id"], row["model_run_id"], row["validation_split_no"]) for row in rows
    }
    assert len(grain) == len(rows) == 11


def test_validation_summary_uses_population_sd_and_prediction_coverage(
    analyst_connection: sqlite3.Connection,
) -> None:
    summaries = {
        row["rate_package_id"]: row
        for row in analyst_connection.execute(
            "SELECT * FROM V_MODEL_VALIDATION_SUMMARY ORDER BY rate_package_id"
        )
    }

    assert set(summaries) == {10, 11, 20, 30, 40, 50}
    for package_id in (10, 11):
        summary = summaries[package_id]
        assert summary["validation_split_count"] == 2
        assert summary["mean_deviance"] == pytest.approx(2.0)
        assert summary["sd_deviance"] == pytest.approx(1.0)
        assert summary["mean_nll"] == pytest.approx(4.0)
        assert summary["sd_nll"] == pytest.approx(2.0)
        assert summary["mean_gini"] == pytest.approx(0.4)
        assert summary["sd_gini"] == pytest.approx(0.2)
        assert summary["validation_prediction_coverage"] == pytest.approx(1.0)
        assert summary["validation_curve_status"] == "COMPLETE"

    assert summaries[11]["validation_evidence"] == "INHERITED_FROM_PARENT"
    assert summaries[11]["validation_source_model_run_id"] == "run-root"
    assert summaries[10]["family"] == "Poisson"
    assert json.loads(summaries[10]["family_params_json"]) == {"dispersion": 1.0}
    assert summaries[10]["link"] == "log"
    assert summaries[20]["validation_split_count"] == 3
    assert summaries[20]["sd_deviance"] == pytest.approx(math.sqrt(8.0 / 3.0))
    assert summaries[20]["validation_prediction_coverage"] == pytest.approx(1.0)
    assert summaries[30]["sd_deviance"] == pytest.approx(0.0)
    assert summaries[30]["validation_prediction_coverage"] == pytest.approx(0.25)
    assert summaries[30]["family"] == "Gaussian"
    assert json.loads(summaries[30]["family_params_json"]) == {}
    assert summaries[30]["link"] == "identity"
    assert summaries[40]["validation_prediction_coverage"] == pytest.approx(0.125)
    assert summaries[50]["validation_prediction_coverage"] == pytest.approx(2.0 / 3.0)
    assert summaries[50]["mean_deviance"] == pytest.approx(12.0)
    assert summaries[50]["sd_deviance"] == pytest.approx(2.0)
    assert summaries[50]["mean_nll"] is None
    assert summaries[50]["sd_nll"] is None
    assert summaries[50]["mean_gini"] is None
    assert summaries[50]["sd_gini"] is None


def test_validation_summary_population_sd_is_stable_for_large_metrics(
    analyst_connection: sqlite3.Connection,
) -> None:
    analyst_connection.execute(
        """
        UPDATE CV_FOLD_METRIC
        SET metric_value = CASE fold_no
            WHEN 1 THEN 1e16
            WHEN 2 THEN 1e16 + 2.0
        END
        WHERE model_run_id = 'run-root'
          AND metric_name IN ('deviance', 'nll', 'gini')
        """
    )

    summary = analyst_connection.execute(
        """
        SELECT validation_split_count, sd_deviance, sd_nll, sd_gini
        FROM V_MODEL_VALIDATION_SUMMARY
        WHERE rate_package_id = 10
        """
    ).fetchone()

    assert summary is not None
    assert summary["validation_split_count"] == 2
    assert summary["sd_deviance"] == pytest.approx(1.0)
    assert summary["sd_nll"] == pytest.approx(1.0)
    assert summary["sd_gini"] == pytest.approx(1.0)


def test_validation_curve_view_is_training_split_evidence_with_link_truth(
    analyst_connection: sqlite3.Connection,
) -> None:
    rows = analyst_connection.execute(
        """
        SELECT *
        FROM V_MODEL_VALIDATION_SPLIT_RELATIVITY
        ORDER BY rate_package_id, validation_split_no, term_name, point_no
        """
    ).fetchall()

    assert len(rows) == 9
    assert {row["model_fit_scope"] for row in rows} == {"VALIDATION_TRAINING_SPLIT_MODEL"}
    assert len(
        {
            (
                row["rate_package_id"],
                row["validation_split_no"],
                row["term_name"],
                row["point_no"],
            )
            for row in rows
        }
    ) == len(rows)

    root_rows = [row for row in rows if row["rate_package_id"] == 10]
    child_rows = [row for row in rows if row["rate_package_id"] == 11]
    identity_rows = [row for row in rows if row["rate_package_id"] == 30]
    assert len(root_rows) == len(child_rows) == 4
    assert all(row["link"] == "log" and row["relativity"] is not None for row in root_rows)
    assert all(
        row["relativity"] == pytest.approx(math.exp(row["eta_contribution"])) for row in root_rows
    )
    assert all(row["validation_evidence"] == "INHERITED_FROM_PARENT" for row in child_rows)
    assert all(row["validation_source_model_run_id"] == "run-root" for row in child_rows)
    assert [(row["term_name"], row["eta_contribution"]) for row in child_rows] == [
        (row["term_name"], row["eta_contribution"]) for row in root_rows
    ]
    assert len(identity_rows) == 1
    assert identity_rows[0]["link"] == "identity"
    assert identity_rows[0]["eta_contribution"] == pytest.approx(2.5)
    assert identity_rows[0]["relativity"] is None

    final_terms = {
        row[0]
        for row in analyst_connection.execute(
            "SELECT DISTINCT term_name FROM V_FINAL_MODEL_RELATIVITY"
        )
    }
    assert "CurveOnly" in {row["term_name"] for row in rows}
    assert "CurveOnly" not in final_terms


def test_current_dataset_view_contains_current_split_geometry_only(
    analyst_connection: sqlite3.Connection,
) -> None:
    rows = analyst_connection.execute(
        """
        SELECT *
        FROM V_CURRENT_DATASET_VALIDATION_SPLIT
        ORDER BY dataset_name, validation_split_no
        """
    ).fetchall()

    assert len(rows) == 9
    assert "manifest-old-kfold" not in {row["manifest_id"] for row in rows}
    assert {
        method: sum(row["split_method"] == method for row in rows)
        for method in {row["split_method"] for row in rows}
    } == {
        "kfold": 2,
        "column_kfold": 3,
        "train_test_split": 1,
        "column_holdout": 1,
        "custom": 2,
    }
    assert {
        "model_id",
        "model_run_id",
        "rate_package_id",
        "deviance",
        "nll",
        "gini",
        "model_fit_scope",
    }.isdisjoint(rows[0].keys())


def _sql_server_view_columns(source: str, view_name: str) -> list[str]:
    match = re.search(
        rf"CREATE OR ALTER VIEW pricing\.{re.escape(view_name)} AS\n"
        rf"SELECT\n(?P<select>.*?)\nFROM ",
        source,
        re.S,
    )
    assert match is not None
    columns: list[str] = []
    for line in match.group("select").splitlines():
        expression = line.strip().rstrip(",")
        alias = re.search(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)$", expression, re.I)
        columns.append(alias.group(1) if alias else expression.split(".")[-1])
    return columns


def test_public_view_names_vocabulary_and_cross_engine_columns_are_exact(
    analyst_connection: sqlite3.Connection,
) -> None:
    migration = Path("db/migrations/V035__clean_validation_evidence_workflow.sql").read_text(
        encoding="utf-8"
    )
    sqlite_views = {
        row["name"]: row["sql"]
        for row in analyst_connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'view' ORDER BY name"
        )
    }

    assert set(sqlite_views) == set(_ANALYST_VIEWS)
    assert {
        "V_MODEL_RELATIVITY",
        "V_PUBLISHED_MODEL_RELATIVITY",
        "V_CURRENT_DATASET_CV_FOLD",
    }.isdisjoint(sqlite_views)
    for view_name, view_sql in sqlite_views.items():
        sqlite_columns = [
            row["name"] for row in analyst_connection.execute(f"PRAGMA table_info('{view_name}')")
        ]
        assert sqlite_columns == _sql_server_view_columns(migration, view_name)
        vocabulary = view_sql.lower()
        assert "pooled" not in vocabulary
        assert "out_of_fold" not in vocabulary
        assert "oof" not in vocabulary
        assert "train_folds_json" not in vocabulary
        assert "test_fold_no" not in vocabulary

        sql_server_view = migration.split(
            f"CREATE OR ALTER VIEW pricing.{view_name} AS\n",
            maxsplit=1,
        )[1].split("\nGO\n", maxsplit=1)[0]
        sql_server_vocabulary = sql_server_view.lower()
        assert "pooled" not in sql_server_vocabulary
        assert "out_of_fold" not in sql_server_vocabulary
        assert "oof" not in sql_server_vocabulary
        assert "train_folds_json" not in sql_server_vocabulary
        assert "test_fold_no" not in sql_server_vocabulary

    for old_view in (
        "V_MODEL_RELATIVITY",
        "V_PUBLISHED_MODEL_RELATIVITY",
        "V_CURRENT_DATASET_CV_FOLD",
    ):
        assert f"DROP VIEW pricing.{old_view}" in migration
        assert f"CREATE OR ALTER VIEW pricing.{old_view}" not in migration
