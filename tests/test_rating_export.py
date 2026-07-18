from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from pricing_pipeline.infra.offline_sqlite import (
    apply_offline_ddl,
    sqlite_engine_with_offline_schemas,
)
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.orchestration import pipeline
from pricing_pipeline.publishing import lineage, rating_export, staging
from pricing_pipeline.publishing.lifecycle import (
    CompletedModelPublishResult,
    PublishResult,
)
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
    write_publication_receipt,
)
from pricing_pipeline.workbench.artifacts import CandidateBundle, save_candidate_bundle


MODEL_CONFIG = ModelBuildConfig(
    model_name="MTPL_FREQ",
    model_label="MTPL frequency",
    target_name="ClaimNb",
    model_type="superglm_poisson",
    deployment_slot="MTPL_FREQ_UAT",
)


class _ExportModel:
    def __init__(self):
        self.calls = []

    def export_rating_tables(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        Path(args[0]).write_bytes(b"workbook")


def test_build_export_id_is_path_safe():
    export_id = rating_export.build_export_id(
        "MTPL_FREQ",
        "scheduled__2026-04-27T10:30:00+00:00",
    )

    assert export_id == "mtpl_freq__scheduled__20260427t1030000000"


def test_export_rating_tables_forwards_weights_and_offset(tmp_path: Path):
    model = _ExportModel()
    output_path = tmp_path / "nested" / "rating_tables.xlsx"
    X = pd.DataFrame({"x": [1, 2]})
    y = np.array([0.0, 1.0])
    export_weight = np.array([0.5, 2.0])
    offset = np.log(np.array([1.0, 3.0]))
    offset_source = pd.Series([12, 36], name="TermMonths")

    result = rating_export.export_rating_tables(
        model,
        X,
        y,
        export_weight,
        output_path,
        offset=offset,
        offset_source=offset_source,
        offset_name="TermMonths",
        offset_kind="discrete",
        offset_max_exact_levels=50,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"
    args, kwargs = model.calls[0]
    assert args == (output_path, X, y)
    np.testing.assert_array_equal(kwargs.pop("sample_weight"), export_weight)
    np.testing.assert_array_equal(kwargs.pop("offset"), offset)
    pd.testing.assert_series_equal(kwargs.pop("offset_source"), offset_source)
    assert kwargs == {
        "n_bins": 150,
        "offset_name": "TermMonths",
        "offset_kind": "discrete",
        "offset_max_exact_levels": 50,
    }


def test_export_rating_tables_preserves_raw_offset_levels(tmp_path: Path):
    from superglm import Categorical, SuperGLM

    row_count = 80
    term = pd.Series(np.resize([12.0, 36.0], row_count), name="Term")
    X = pd.DataFrame({"territory": np.resize(["north", "south"], row_count)})
    offset = np.log(term / 12.0)
    y = np.random.default_rng(20260716).poisson(np.exp(-1.0 + offset.to_numpy()))
    model = SuperGLM(
        features={"territory": Categorical(base="first")},
        selection_penalty=0.0,
    ).fit(X, y, offset=offset)
    workbook_path = tmp_path / "rating_tables.xlsx"

    rating_export.export_rating_tables(
        model,
        X,
        y,
        np.ones(row_count),
        workbook_path,
        offset=offset,
        offset_source=term,
        offset_name="Term",
        offset_kind="discrete",
    )

    raw = pd.read_excel(workbook_path, sheet_name="Rating Tables", header=None)
    header_positions = [
        (row, column)
        for row in range(len(raw))
        for column in range(len(raw.columns) - 1)
        if raw.iat[row, column] == "Term" and raw.iat[row, column + 1] == "Relativity"
    ]
    assert len(header_positions) == 1
    header_row, source_column = header_positions[0]
    offset_rows = raw.iloc[header_row + 1 :, [source_column, source_column + 1]].dropna()
    relativities = dict(
        zip(
            pd.to_numeric(offset_rows.iloc[:, 0]),
            pd.to_numeric(offset_rows.iloc[:, 1]),
            strict=True,
        )
    )

    assert relativities == pytest.approx({12.0: 1.0, 36.0: 3.0})


def test_export_rating_tables_requires_supported_superglm(tmp_path: Path):
    with pytest.raises(RuntimeError, match=r"SuperGLM.*export_rating_tables"):
        rating_export.export_rating_tables(
            object(),
            pd.DataFrame({"x": [1]}),
            np.array([0.0]),
            np.array([1.0]),
            tmp_path / "rating_tables.xlsx",
        )


def _offline_staging_engine(tmp_path: Path):
    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    return engine


def _minimal_rating_workbook(
    path: Path,
    *,
    term_name: str = "TermMonths",
    level_code: str = "12",
) -> Path:
    raw = pd.DataFrame([[None] * 3 for _ in range(8)])
    raw.iat[1, 2] = 0.123
    raw.iat[4, 0] = term_name
    raw.iloc[6, 0:3] = [term_name, "Relativity", "Weight"]
    raw.iloc[7, 0:3] = [level_code, 1.0, 10.0]
    raw.to_excel(path, sheet_name="Rating Tables", header=False, index=False)
    return path


def _publication_receipt(
    *,
    handling: str = "EXPORTED_FACTOR",
    published_factor_name: str | None = "TermMonths",
    term_metadata: dict[str, dict[str, object]] | None = None,
) -> SuperGLMPublicationReceipt:
    if handling == "EXPORTED_FACTOR":
        offset_contract = OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="term_months",
            published_factor_name=published_factor_name,
            source_name="Exposure",
            label="Policy exposure term",
        )
        default_term_metadata = {
            "TermMonths": {
                "feature_kind": "offset",
                "source_column": "Exposure",
            }
        }
    else:
        offset_contract = OffsetExportContract(handling="NONE")
        default_term_metadata = {}

    return SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version="1.0.0",
        extractor_version="unit-test",
        package_metadata={"model": {"family": "poisson", "link": "log"}},
        term_metadata=term_metadata if term_metadata is not None else default_term_metadata,
        offset_contract=offset_contract,
    )


def test_build_staging_frames_accepts_mapping_and_uses_standard_layout(tmp_path: Path):
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")
    args = staging.StagingExport(
        workbook_path=workbook_path,
        export_id="export-1",
        model_name="MTPL_FREQ",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_version="v1",
        effective_from=None,
        effective_to=None,
        interaction_features=MappingProxyType({}),
        created_by="analyst@example.test",
        replace=False,
        model_id=17,
    )

    export, rates, _levels = staging.build_staging_frames(args)

    assert export.iloc[0]["base_rate"] == pytest.approx(0.123)
    assert export.iloc[0]["source_file"] == str(workbook_path.resolve())
    assert rates.iloc[0]["term_name"] == "TermMonths"


def test_stage_rating_export_persists_receipt_offset_and_content_digest(tmp_path: Path):
    engine = _offline_staging_engine(tmp_path)
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")
    receipt_path = tmp_path / "publication_receipt.json"
    receipt_sha256 = write_publication_receipt(_publication_receipt(), receipt_path)

    content_sha256 = staging.stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id="export-1",
        expected_database=None,
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from=None,
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=receipt_sha256,
        replace=True,
        model_id=1,
    )

    with engine.begin() as connection:
        export = (
            connection.execute(
                text(
                    "SELECT publication_receipt_sha256, offset_handling, "
                    "offset_factor_name, staging_content_sha256 "
                    "FROM pricing_stg.STG_RATING_EXPORT WHERE export_id = :export_id"
                ),
                {"export_id": "export-1"},
            )
            .mappings()
            .one()
        )
        term = (
            connection.execute(
                text(
                    "SELECT term_type FROM pricing_stg.STG_RATE_CELL "
                    "WHERE export_id = :export_id AND term_name = :term_name"
                ),
                {"export_id": "export-1", "term_name": "TermMonths"},
            )
            .mappings()
            .one()
        )
        metadata_json = connection.execute(
            text(
                "SELECT term_metadata_json FROM pricing_stg.STG_TERM_METADATA "
                "WHERE export_id = :export_id AND term_name = :term_name"
            ),
            {"export_id": "export-1", "term_name": "TermMonths"},
        ).scalar_one()

    assert len(content_sha256) == 64
    assert export["publication_receipt_sha256"] == receipt_sha256
    assert export["staging_content_sha256"] == content_sha256
    assert export["offset_handling"] == "EXPORTED_FACTOR"
    assert export["offset_factor_name"] == "TermMonths"
    assert term["term_type"] == "OFFSET_FACTOR"
    assert json.loads(metadata_json)["feature_kind"] == "offset"


def test_insert_staging_frames_rechecks_database_before_mutation(monkeypatch):
    class WrongDatabaseConnection:
        def __init__(self):
            self.statements = []

        def execute(self, statement, params=None):
            sql = str(statement)
            self.statements.append((sql, params))
            if "DB_NAME()" in sql:
                return _Result(scalar="OtherDb")
            raise AssertionError("database guard must run before staging SQL")

    class Begin:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, exc, tb):
            return False

    class Engine:
        def __init__(self):
            self.connection = WrongDatabaseConnection()

        def begin(self):
            return Begin(self.connection)

    engine = Engine()
    monkeypatch.setattr(
        staging,
        "schema_names_from_connectable",
        lambda _engine: SimpleNamespace(pricing_staging="pricing_stg"),
    )
    args = staging.StagingExport(
        workbook_path=Path("rating.xlsx"),
        export_id="export-1",
        model_name="MTPL_FREQ",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_version="v1",
        effective_from=None,
        effective_to=None,
        interaction_features={},
        created_by="airflow",
        replace=False,
        model_id=17,
    )

    with pytest.raises(RuntimeError, match="expected 'PricingLab'.*connected to 'OtherDb'"):
        staging.insert_staging_frames(
            engine,
            args,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            expected_database="PricingLab",
        )

    assert len(engine.connection.statements) == 1
    assert "DB_NAME()" in engine.connection.statements[0][0]


def test_staging_database_guard_allows_missing_target_only_for_sqlite():
    remote_connection = SimpleNamespace(dialect=SimpleNamespace(name="mssql"))
    sqlite_connection = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

    with pytest.raises(ValueError, match="expected_database is required"):
        staging._verify_expected_database(remote_connection, None)

    staging._verify_expected_database(sqlite_connection, None)


def test_stage_rating_export_requires_publication_receipt(tmp_path: Path):
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")
    with pytest.raises(TypeError, match="publication_receipt"):
        staging.stage_rating_export(
            object(),
            workbook_path=workbook_path,
            export_id="export-1",
            expected_database=None,
            model_name="MTPL_FREQ",
            model_version="v1",
            effective_from=None,
            model_id=1,
        )


def test_stage_rating_export_rejects_receipt_term_missing_from_workbook(tmp_path: Path):
    workbook_path = _minimal_rating_workbook(
        tmp_path / "rating_tables.xlsx",
        term_name="LogDensity",
        level_code="per_unit",
    )
    receipt_path = tmp_path / "publication_receipt.json"
    receipt_sha256 = write_publication_receipt(
        _publication_receipt(
            handling="NONE",
            term_metadata={
                "LogDensity": {"feature_kind": "numeric"},
                "MissingTerm": {"feature_kind": "categorical"},
            },
        ),
        receipt_path,
    )

    with pytest.raises(ValueError, match="not present in staged workbook"):
        staging.stage_rating_export(
            object(),
            workbook_path=workbook_path,
            export_id="export-1",
            expected_database=None,
            model_name="MTPL_FREQ",
            model_version="v1",
            effective_from=None,
            publication_receipt_path=receipt_path,
            publication_receipt_sha256=receipt_sha256,
            model_id=1,
        )


def test_staging_content_sha256_binds_every_frame():
    export = pd.DataFrame([{"export_id": "export-1", "model_name": "MTPL_FREQ"}])
    rates = pd.DataFrame(
        [{"export_id": "export-1", "row_id": 1, "term_name": "Area", "multiplier": 1.1}]
    )
    levels = pd.DataFrame(
        [{"export_id": "export-1", "row_id": 1, "position_no": 1, "level_code": "A"}]
    )
    terms = pd.DataFrame(
        [
            {
                "export_id": "export-1",
                "term_name": "Area",
                "term_metadata_json": '{"feature_kind":"categorical"}',
            }
        ]
    )
    digest = staging.staging_content_sha256(export, rates, levels, terms)

    changed_rates = rates.copy()
    changed_rates.loc[0, "multiplier"] = 1.2
    changed_levels = levels.copy()
    changed_levels.loc[0, "level_code"] = "B"
    changed_terms = terms.copy()
    changed_terms.loc[0, "term_metadata_json"] = '{"feature_kind":"numeric"}'

    assert len(digest) == 64
    assert staging.staging_content_sha256(export, changed_rates, levels, terms) != digest
    assert staging.staging_content_sha256(export, rates, changed_levels, terms) != digest
    assert staging.staging_content_sha256(export, rates, levels, changed_terms) != digest


def test_stage_rating_export_parses_categorical_interaction_matrix(
    monkeypatch,
    tmp_path: Path,
):
    from superglm import Categorical, SuperGLM

    from pricing_pipeline.publishing.superglm_metadata import (
        build_superglm_publication_receipt,
    )

    n_rows = 60
    X = pd.DataFrame(
        {
            "territory": np.resize(["urban", "rural"], n_rows),
            "age_band": np.resize(["young", "old", "mid"], n_rows),
        }
    )
    y = np.random.default_rng(20260713).poisson(0.4, size=n_rows)
    weights = np.ones(n_rows)
    model = SuperGLM(
        features={
            "territory": Categorical(base="first"),
            "age_band": Categorical(base="first"),
        },
        interactions=[("territory", "age_band")],
        selection_penalty=0.0,
    ).fit(X, y, sample_weight=weights)
    workbook_path = tmp_path / "rating_tables.xlsx"
    rating_export.export_rating_tables(
        model,
        X,
        y,
        weights,
        workbook_path,
    )
    receipt_path = tmp_path / "publication_receipt.json"
    receipt_sha256 = write_publication_receipt(
        build_superglm_publication_receipt(
            model,
            offset_contract=OffsetExportContract(handling="NONE"),
        ),
        receipt_path,
    )
    captured = {}
    monkeypatch.setattr(
        staging,
        "insert_staging_frames",
        lambda engine, args, export, rates, levels, terms, **kwargs: captured.update(
            rates=rates.copy(),
            levels=levels.copy(),
            terms=terms.copy(),
        ),
    )

    staging.stage_rating_export(
        object(),
        workbook_path=workbook_path,
        export_id="export-1",
        expected_database=None,
        model_name="MTPL_FREQ",
        model_version="v1",
        effective_from=None,
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=receipt_sha256,
        model_id=17,
    )

    interactions = captured["rates"].query("term_name == 'territory_age_band'")
    interaction_levels = captured["levels"].loc[
        captured["levels"]["row_id"].isin(interactions["row_id"])
    ]
    assert len(interactions) == 6
    assert set(interactions["term_type"]) == {"CATEGORICAL_INTERACTION"}
    assert len(interaction_levels) == 12
    assert set(interaction_levels["position_no"]) == {1, 2}
    assert set(captured["terms"]["term_name"]) == {
        "territory",
        "age_band",
        "territory_age_band",
    }


class _Result:
    def __init__(self, *, row=None, rows=(), scalar=None):
        self.row = row
        self.rows = list(rows)
        self.scalar = scalar

    def mappings(self):
        return self

    def one_or_none(self):
        return self.row

    def all(self):
        return self.rows

    def scalar_one(self):
        return self.scalar

    def scalar_one_or_none(self):
        return self.scalar


class _LineageConnection:
    def __init__(
        self,
        *,
        existing=None,
        associations=(),
        parent_validation_source_model_run_id=409,
    ):
        self.existing = existing
        self.associations = list(associations)
        self.parent_validation_source_model_run_id = parent_validation_source_model_run_id
        self.statements = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append((sql, params))
        if "FROM pricing.PRICING_RATE_PACKAGE AS package" in sql:
            return _Result(
                row={
                    "model_id": 17,
                    "model_version": "v1",
                    "source_export_id": "export-1",
                    "parent_rate_package_id": 41,
                    "build_fingerprint_sha256": None,
                }
            )
        if "FROM pricing.CV_SPLIT_SET AS split_set" in sql:
            return _Result(
                rows=[
                    {
                        "manifest_id": "manifest-1",
                        "fold_no": 1,
                        "n_train": 1,
                        "n_test": 1,
                    }
                ]
            )
        if "FROM pricing.MODEL_RUN AS mr" in sql:
            return _Result(row=self.existing)
        if "lineage_source" in sql:
            return _Result(rows=self.associations)
        if "child_package.parent_rate_package_id" in sql:
            return _Result(scalar=self.parent_validation_source_model_run_id)
        if "SELECT model_run_id" in sql:
            return _Result(scalar=501)
        return _Result()


class _Begin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _Engine:
    def __init__(self, connection):
        self.connection = connection

    def begin(self):
        return _Begin(self.connection)


def _model_run_build(**overrides):
    values = {
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "v1",
        "model_type": "superglm_poisson",
        "target_name": "ClaimNb",
        "deployment_slot": "MTPL_FREQ_UAT",
        "manifest_id": "manifest-1",
        "split_set_id": "split-1",
        "export_id": "export-1",
        "rating_workbook_path": "/tmp/rating.xlsx",
        "rating_workbook_sha256": "f" * 64,
        "created_by": "analyst@example.test",
        "mlflow_run_id": None,
        "publication_receipt_path": "/tmp/publication_receipt.json",
        "publication_receipt_sha256": "c" * 64,
        "candidate_artifact_path": "/tmp/candidate.joblib",
        "candidate_artifact_sha256": "a" * 64,
        "candidate_artifact_format": "superglm-candidate-joblib-v3",
        "candidate_artifact_size_bytes": 123,
        "candidate_python_version": "3.14.4",
        "candidate_superglm_version": "0.12.0",
        "candidate_superglm_git_sha": "f" * 40,
        "build_fingerprint_sha256": "0" * 64,
        "builder_source_sha256": "1" * 64,
        "materialized_split_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "candidate_superglm_sha256": "4" * 64,
        "row_order_sha256": "5" * 64,
        "model_source_sha256": "b" * 64,
        "model_frame_sha256": "d" * 64,
    }
    values.update(overrides)
    return ModelExportResult(**values)


def _model_run_kwargs():
    return {
        "build": _model_run_build(),
        "dag_id": "pricing.model.build",
        "airflow_run_id": "notebook__2026-07-12",
        "rate_package_id": 42,
        "parent_model_run_id": 409,
    }


def _model_run_row():
    kwargs = _model_run_kwargs()
    build = kwargs.pop("build")
    return {
        "model_run_id": 501,
        **kwargs,
        "mlflow_run_id": build.mlflow_run_id,
        "manifest_id": build.manifest_id,
        "export_id": build.export_id,
        "model_id": build.model_id,
        "model_name": build.model_name,
        "model_version": build.model_version,
        "rating_workbook_path": build.rating_workbook_path,
        "rating_workbook_sha256": build.rating_workbook_sha256,
        "run_status": "SUCCESS",
        "created_by": build.created_by,
        "publication_receipt_path": build.publication_receipt_path,
        "publication_receipt_sha256": build.publication_receipt_sha256,
        "candidate_artifact_path": build.candidate_artifact_path,
        "candidate_artifact_sha256": build.candidate_artifact_sha256,
        "candidate_artifact_format": build.candidate_artifact_format,
        "candidate_artifact_size_bytes": build.candidate_artifact_size_bytes,
        "candidate_python_version": build.candidate_python_version,
        "candidate_superglm_version": build.candidate_superglm_version,
        "candidate_superglm_git_sha": build.candidate_superglm_git_sha,
        "model_source_sha256": build.model_source_sha256,
        "builder_source_sha256": build.builder_source_sha256,
        "materialized_split_sha256": build.materialized_split_sha256,
        "runtime_sha256": build.runtime_sha256,
        "candidate_superglm_sha256": build.candidate_superglm_sha256,
        "validation_curve_status": build.validation_curve_status,
        "validation_curve_reason": build.validation_curve_reason,
        "validation_source_model_run_id": 409,
    }


def _model_run_associations():
    return [
        {
            "lineage_source": "actual_dataset",
            "manifest_id": "manifest-1",
            "split_set_id": None,
            "dataset_role": "training",
            "split_role": None,
        },
        {
            "lineage_source": "actual_dataset",
            "manifest_id": "parent-manifest",
            "split_set_id": None,
            "dataset_role": "training",
            "split_role": None,
        },
        {
            "lineage_source": "parent_dataset",
            "manifest_id": "parent-manifest",
            "split_set_id": None,
            "dataset_role": "training",
            "split_role": None,
        },
        {
            "lineage_source": "actual_split",
            "manifest_id": "manifest-1",
            "split_set_id": "split-1",
            "dataset_role": "training",
            "split_role": "validation",
        },
        {
            "lineage_source": "actual_split",
            "manifest_id": "parent-manifest",
            "split_set_id": "parent-split",
            "dataset_role": "training",
            "split_role": "benchmark",
        },
        {
            "lineage_source": "parent_split",
            "manifest_id": "parent-manifest",
            "split_set_id": "parent-split",
            "dataset_role": "training",
            "split_role": "benchmark",
        },
    ]


def test_record_model_run_writes_identity_associations_parent_and_metrics():
    connection = _LineageConnection()
    kwargs = _model_run_kwargs()
    kwargs["build"] = _model_run_build(
        metrics={"deviance": 0.42},
        metric_scopes={"deviance": "cv"},
        fold_metrics=({"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},),
    )

    model_run_id = lineage.record_model_run(
        _Engine(connection),
        **kwargs,
    )

    assert model_run_id == 501
    model_run_write = next(
        item for item in connection.statements if "MERGE pricing.MODEL_RUN" in item[0]
    )
    assert model_run_write[1]["candidate_artifact_sha256"] == "a" * 64
    assert model_run_write[1]["model_source_sha256"] == "b" * 64
    assert model_run_write[1]["parent_model_run_id"] == 409
    assert any("MERGE mlops.MODEL_RUN_DATASET" in sql for sql, _ in connection.statements)
    assert any("MERGE mlops.MODEL_RUN_SPLIT_SET" in sql for sql, _ in connection.statements)
    assert any("MERGE mlops.MODEL_RUN_METRIC" in sql for sql, _ in connection.statements)
    assert any("MERGE pricing.CV_FOLD_METRIC" in sql for sql, _ in connection.statements)
    assert any(
        "parent_dataset.model_run_id = :parent_model_run_id" in sql
        for sql, _ in connection.statements
    )


def test_record_model_run_exact_retry_is_read_only():
    connection = _LineageConnection(
        existing=_model_run_row(),
        associations=_model_run_associations(),
    )

    model_run_id = lineage.record_model_run(
        None,
        connection=connection,
        **_model_run_kwargs(),
    )

    assert model_run_id == 501
    assert not any(
        sql.lstrip().startswith(("MERGE", "DELETE", "INSERT", "UPDATE"))
        for sql, _ in connection.statements
    )


def test_record_model_run_retry_rejects_changed_validation_source():
    existing = _model_run_row()
    existing["validation_source_model_run_id"] = 410
    connection = _LineageConnection(
        existing=existing,
        associations=_model_run_associations(),
        parent_validation_source_model_run_id=409,
    )

    with pytest.raises(
        lineage.ModelRunIdentityError,
        match="validation_source_model_run_id",
    ):
        lineage.record_model_run(
            None,
            connection=connection,
            **_model_run_kwargs(),
        )

    assert not any(
        sql.lstrip().startswith(("MERGE", "DELETE", "INSERT", "UPDATE"))
        for sql, _ in connection.statements
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("manifest_id", "manifest-2"),
        ("model_version", "v2"),
        ("candidate_artifact_sha256", "c" * 64),
        ("candidate_superglm_git_sha", "e" * 40),
        ("rating_workbook_sha256", "c" * 64),
        ("parent_model_run_id", 410),
    ],
)
def test_record_model_run_rejects_changed_immutable_retry(field_name, changed_value):
    connection = _LineageConnection(existing=_model_run_row())
    kwargs = _model_run_kwargs()
    if field_name == "parent_model_run_id":
        kwargs[field_name] = changed_value
    else:
        kwargs["build"] = kwargs["build"].model_copy(update={field_name: changed_value})

    with pytest.raises(lineage.ModelRunIdentityError, match=field_name):
        lineage.record_model_run(None, connection=connection, **kwargs)

    assert not any(
        sql.lstrip().startswith(("MERGE", "DELETE", "INSERT", "UPDATE"))
        for sql, _ in connection.statements
    )


_RETRY_ARTIFACTS = {}


def _retry_artifact_fields(tmp_path: Path):
    key = tmp_path.resolve()
    if key not in _RETRY_ARTIFACTS:
        receipt = tmp_path / "publication_receipt.json"
        receipt.write_bytes(b"publication receipt")
        metadata = save_candidate_bundle(
            CandidateBundle(
                fitted_model=SimpleNamespace(name="candidate"),
                X=pd.DataFrame({"age": [42.0]}),
                y=np.zeros(1),
                sample_weight=None,
                offset=None,
                export_weight=None,
                cv_report={},
                model_name="MTPL_FREQ",
                model_version="v1",
                export_id="export-1",
                manifest_id="manifest-1",
                split_set_id="split-1",
                pk_columns=("policy_id",),
                row_order_sha256="d" * 64,
                model_source_sha256="c" * 64,
                model_frame_sha256="e" * 64,
                build_fingerprint_sha256="1" * 64,
                builder_source_sha256="2" * 64,
                materialized_split_sha256="3" * 64,
                runtime_sha256="4" * 64,
                candidate_superglm_sha256="5" * 64,
                offset_contract={"handling": "NONE"},
            ),
            tmp_path / "candidate.joblib",
        )
        _RETRY_ARTIFACTS[key] = {
            "publication_receipt_path": str(receipt),
            "publication_receipt_sha256": pipeline.sha256_file(receipt),
            "candidate_artifact_path": str(metadata.path),
            "candidate_artifact_sha256": metadata.sha256,
            "candidate_artifact_format": metadata.format,
            "candidate_artifact_size_bytes": metadata.size_bytes,
            "candidate_python_version": metadata.python_version,
            "candidate_superglm_version": metadata.superglm_version,
            "candidate_superglm_git_sha": metadata.superglm_git_sha,
            "build_fingerprint_sha256": "1" * 64,
            "builder_source_sha256": "2" * 64,
            "materialized_split_sha256": "3" * 64,
            "runtime_sha256": "4" * 64,
            "candidate_superglm_sha256": "5" * 64,
            "row_order_sha256": "d" * 64,
            "model_source_sha256": "c" * 64,
            "model_frame_sha256": "e" * 64,
        }
    return _RETRY_ARTIFACTS[key]


def _retry_export(tmp_path: Path) -> ModelExportResult:
    workbook_path = (tmp_path / "rating_tables.xlsx").resolve()
    if not workbook_path.exists():
        workbook_path.write_bytes(b"rating workbook")
    return ModelExportResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="v1",
        model_type="superglm_poisson",
        target_name="ClaimNb",
        deployment_slot="MTPL_FREQ_UAT",
        manifest_id="manifest-1",
        mlflow_run_id=None,
        split_set_id="split-1",
        export_id="export-1",
        rating_workbook_path=str(workbook_path),
        rating_workbook_sha256=pipeline.sha256_file(workbook_path),
        effective_from=None,
        created_by="analyst@example.test",
        metrics={"deviance": 0.42},
        metric_scopes={"deviance": "cv"},
        fold_metrics=({"fold_no": 1, "metric_name": "deviance", "metric_value": 0.4},),
        validation_splits=(
            {
                "validation_split_no": 1,
                "n_train": 1,
                "n_validation": 1,
                "metrics": {"deviance": 0.4},
            },
        ),
        validation_curve_status="UNAVAILABLE",
        validation_curve_reason="curve comparison unavailable",
        **_retry_artifact_fields(tmp_path),
    )


def _write_timestamped_xlsx(path: Path, *, value: str, year: int) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active["A1"] = value
    workbook.properties.created = datetime(year, 1, 1, tzinfo=UTC)
    workbook.properties.modified = datetime(year, 1, 2, tzinfo=UTC)
    workbook.save(path)
    workbook.close()


def _retry_evidence(tmp_path: Path):
    workbook = (tmp_path / "rating_tables.xlsx").resolve()
    if not workbook.exists():
        workbook.write_bytes(b"rating workbook")
    workbook_path = str(workbook)
    workbook_sha256 = pipeline.sha256_file(workbook)
    artifacts = _retry_artifact_fields(tmp_path)
    return {
        "row": {
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_version": "v1",
            "source_export_id": "export-1",
            "rate_package_id": 42,
            "package_version": 3,
            "package_status": "PUBLISHED",
            "parent_rate_package_id": None,
            "effective_from_date": None,
            "effective_to_date": None,
            "source_file": workbook_path,
            "package_publication_receipt_sha256": artifacts["publication_receipt_sha256"],
            "model_run_id": 501,
            "run_status": "SUCCESS",
            "run_export_id": "export-1",
            "run_model_id": 17,
            "run_model_name": "MTPL_FREQ",
            "run_model_version": "v1",
            "dag_id": "notebook",
            "airflow_run_id": "export-1",
            "manifest_id": "manifest-1",
            "rating_workbook_path": workbook_path,
            "rating_workbook_sha256": workbook_sha256,
            "mlflow_run_id": None,
            "validation_curve_status": "UNAVAILABLE",
            "validation_curve_reason": "curve comparison unavailable",
            "validation_source_model_run_id": 501,
            **artifacts,
        },
        "datasets": [{"manifest_id": "manifest-1", "dataset_role": "training"}],
        "splits": [
            {
                "manifest_id": "manifest-1",
                "split_set_id": "split-1",
                "dataset_role": "training",
                "split_role": "validation",
            }
        ],
        "metrics": [{"metric_name": "deviance", "metric_value": 0.42, "metric_scope": "cv"}],
        "folds": [
            {
                "split_set_id": "split-1",
                "fold_no": 1,
                "metric_name": "deviance",
                "metric_value": 0.4,
            }
        ],
        "curve_points": [],
        "manifests": {
            "manifest-1": {
                "manifest_id": "manifest-1",
                "dataset_name": "mtpl",
                "source_system": "unit-test",
                "data_as_of_date": "2026-07-12",
                "row_count": 1,
                "pk_columns_json": '["policy_id"]',
                "target_column": "ClaimNb",
                "weight_column": None,
                "exposure_column": None,
                "data_as_of_column": None,
                "model_frame_sha256": artifacts["model_frame_sha256"],
                "frame_hash_metadata_json": '{"algorithm":"sha256"}',
                "offset_column": None,
                "offset_source_column": None,
                "offset_label": None,
                "export_weight_column": None,
            }
        },
        "columns": {
            "manifest-1": [
                {
                    "ordinal_no": 1,
                    "column_name": "policy_id",
                    "column_role": "KEY",
                    "pandas_dtype": "int64",
                    "null_count": 0,
                    "distinct_count": 1,
                },
                {
                    "ordinal_no": 2,
                    "column_name": "ClaimNb",
                    "column_role": "TARGET",
                    "pandas_dtype": "float64",
                    "null_count": 0,
                    "distinct_count": 1,
                },
            ]
        },
        "split_sets": {
            "split-1": {
                "split_set_id": "split-1",
                "manifest_id": "manifest-1",
                "split_mode": "MATERIALIZED",
                "splitter_class": "unit.Splitter",
                "splitter_params_json": '{"n_splits":1}',
                "row_order_sha256": artifacts["row_order_sha256"],
                "row_count": 1,
                "fold_count": 1,
                "groups_column": None,
                "stratify_column": None,
                "artifact_sha256": artifacts["materialized_split_sha256"],
                "runtime_metadata_json": '{"runtime":"test"}',
            }
        },
        "split_geometry": {"split-1": [{"fold_no": 1, "n_train": 1, "n_test": 1}]},
    }


class _EvidenceConnection:
    def __init__(self, evidence):
        self.evidence = evidence

    def execute(self, statement, params):
        sql = str(statement)
        if "FROM pricing.PRICING_RATE_PACKAGE AS rp" in sql:
            return _Result(rows=[self.evidence["row"]])
        if "FROM mlops.MODEL_RUN_DATASET" in sql:
            return _Result(rows=self.evidence["datasets"])
        if "FROM mlops.MODEL_RUN_SPLIT_SET" in sql:
            return _Result(rows=self.evidence["splits"])
        if "FROM mlops.MODEL_RUN_METRIC" in sql:
            return _Result(rows=self.evidence["metrics"])
        if "FROM pricing.CV_FOLD_METRIC" in sql:
            return _Result(rows=self.evidence["folds"])
        if "FROM pricing.CV_SPLIT_CURVE_POINT" in sql:
            return _Result(rows=self.evidence["curve_points"])
        if "FROM pricing.DATASET_MANIFEST" in sql:
            return _Result(row=self.evidence["manifests"].get(str(params["manifest_id"])))
        if "FROM pricing.DATASET_COLUMN" in sql:
            return _Result(rows=self.evidence["columns"].get(str(params["manifest_id"]), []))
        if "FROM pricing.CV_SPLIT_SET" in sql:
            return _Result(row=self.evidence["split_sets"].get(str(params["split_set_id"])))
        if "FROM pricing.CV_FOLD AS fold" in sql:
            return _Result(
                rows=self.evidence["split_geometry"].get(str(params["split_set_id"]), [])
            )
        raise AssertionError(f"unexpected evidence query: {sql}")


def _change_canonical_manifest_contract(evidence):
    canonical = deepcopy(evidence["manifests"]["manifest-1"])
    canonical.update(manifest_id="canonical-manifest", row_count=2)
    evidence["manifests"]["canonical-manifest"] = canonical
    evidence["columns"]["canonical-manifest"] = deepcopy(evidence["columns"]["manifest-1"])
    evidence["row"]["manifest_id"] = "canonical-manifest"
    evidence["datasets"][0]["manifest_id"] = "canonical-manifest"
    evidence["splits"][0]["manifest_id"] = "canonical-manifest"


def _change_canonical_manifest_columns(evidence):
    canonical = deepcopy(evidence["manifests"]["manifest-1"])
    canonical["manifest_id"] = "canonical-manifest"
    evidence["manifests"]["canonical-manifest"] = canonical
    canonical_columns = deepcopy(evidence["columns"]["manifest-1"])
    canonical_columns[0]["null_count"] = 1
    evidence["columns"]["canonical-manifest"] = canonical_columns
    evidence["row"]["manifest_id"] = "canonical-manifest"
    evidence["datasets"][0]["manifest_id"] = "canonical-manifest"
    evidence["splits"][0]["manifest_id"] = "canonical-manifest"


def _change_canonical_split_geometry(evidence):
    canonical = deepcopy(evidence["split_sets"]["split-1"])
    canonical["split_set_id"] = "canonical-split"
    evidence["split_sets"]["canonical-split"] = canonical
    evidence["split_geometry"]["canonical-split"] = [{"fold_no": 1, "n_train": 1, "n_test": 2}]
    evidence["splits"][0]["split_set_id"] = "canonical-split"


def test_existing_published_run_requires_exact_complete_evidence(tmp_path: Path):
    export = _retry_export(tmp_path)
    evidence = _retry_evidence(tmp_path)

    result = pipeline._resolve_existing_published_run(
        _Engine(_EvidenceConnection(evidence)),
        export,
        allowed_artifact_root=tmp_path,
    )

    assert result == CompletedModelPublishResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="v1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        package_status="PUBLISHED",
        rating_workbook_path=export.rating_workbook_path,
        model_run_id=501,
        mlflow_run_id=None,
        publication_receipt_path=export.publication_receipt_path,
        publication_receipt_sha256=export.publication_receipt_sha256,
        candidate_artifact_path=export.candidate_artifact_path,
        was_existing=True,
    )
    assert result.candidate_artifact_path == export.candidate_artifact_path

    evidence["row"]["model_run_id"] = None
    with pytest.raises(pipeline.PublishedRunIntegrityError, match="manual repair"):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


def test_existing_published_run_accepts_workbook_with_only_generated_timestamp_changes(
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    incoming_workbook = Path(export.rating_workbook_path)
    _write_timestamped_xlsx(incoming_workbook, value="same rating data", year=2025)
    export = export.model_copy(
        update={"rating_workbook_sha256": pipeline.sha256_file(incoming_workbook)}
    )
    evidence = _retry_evidence(tmp_path)
    canonical_workbook = tmp_path / "canonical-rating.xlsx"
    _write_timestamped_xlsx(canonical_workbook, value="same rating data", year=2026)
    evidence["row"].update(
        source_file=str(canonical_workbook),
        rating_workbook_path=str(canonical_workbook),
        rating_workbook_sha256=pipeline.sha256_file(canonical_workbook),
    )
    assert evidence["row"]["rating_workbook_sha256"] != export.rating_workbook_sha256

    result = pipeline._resolve_existing_published_run(
        _Engine(_EvidenceConnection(evidence)),
        export,
        allowed_artifact_root=tmp_path,
    )

    assert result.rating_workbook_path == str(canonical_workbook)


def test_existing_published_run_rejects_semantically_different_workbook(
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    incoming_workbook = Path(export.rating_workbook_path)
    _write_timestamped_xlsx(incoming_workbook, value="incoming rating data", year=2025)
    export = export.model_copy(
        update={"rating_workbook_sha256": pipeline.sha256_file(incoming_workbook)}
    )
    evidence = _retry_evidence(tmp_path)
    canonical_workbook = tmp_path / "canonical-rating.xlsx"
    _write_timestamped_xlsx(canonical_workbook, value="different rating data", year=2026)
    evidence["row"].update(
        source_file=str(canonical_workbook),
        rating_workbook_path=str(canonical_workbook),
        rating_workbook_sha256=pipeline.sha256_file(canonical_workbook),
    )

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="rating workbook semantic content differs",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


@pytest.mark.parametrize("corrupt_side", ["canonical", "incoming"])
def test_existing_published_run_wraps_corrupt_workbook_semantic_verification(
    tmp_path: Path,
    corrupt_side: str,
):
    export = _retry_export(tmp_path)
    incoming_workbook = Path(export.rating_workbook_path)
    if corrupt_side == "incoming":
        incoming_workbook.write_bytes(b"corrupt incoming workbook")
    else:
        _write_timestamped_xlsx(incoming_workbook, value="rating data", year=2025)
    export = export.model_copy(
        update={"rating_workbook_sha256": pipeline.sha256_file(incoming_workbook)}
    )
    evidence = _retry_evidence(tmp_path)
    canonical_workbook = tmp_path / "canonical-rating.xlsx"
    if corrupt_side == "canonical":
        canonical_workbook.write_bytes(b"corrupt canonical workbook")
    else:
        _write_timestamped_xlsx(canonical_workbook, value="rating data", year=2026)
    evidence["row"].update(
        source_file=str(canonical_workbook),
        rating_workbook_path=str(canonical_workbook),
        rating_workbook_sha256=pipeline.sha256_file(canonical_workbook),
    )

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="rating workbook semantic verification failed",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


def test_existing_published_run_ignores_attempt_metadata_and_returns_canonical_paths(
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    evidence = _retry_evidence(tmp_path)
    canonical_attempt = tmp_path / "canonical" / "attempt"
    canonical_attempt.mkdir(parents=True)
    canonical_workbook = canonical_attempt / "rating_tables.xlsx"
    canonical_workbook.write_bytes(Path(export.rating_workbook_path).read_bytes())
    canonical_receipt = canonical_attempt / "publication_receipt.json"
    canonical_receipt.write_bytes(Path(export.publication_receipt_path).read_bytes())
    canonical_candidate = canonical_attempt / "candidate.joblib"
    canonical_candidate.write_bytes(Path(export.candidate_artifact_path).read_bytes())
    evidence["row"].update(
        source_file=str(canonical_workbook),
        rating_workbook_path=str(canonical_workbook),
        publication_receipt_path=str(canonical_receipt),
        candidate_artifact_path=str(canonical_candidate),
        dag_id="another-dag",
        airflow_run_id="another-attempt",
        created_by="another-actor",
    )

    result = pipeline._resolve_existing_published_run(
        _Engine(_EvidenceConnection(evidence)),
        export,
        rate_package_id=42,
        allowed_artifact_root=tmp_path,
    )

    assert result.rating_workbook_path == str(canonical_workbook)
    assert result.publication_receipt_path == str(canonical_receipt)
    assert result.candidate_artifact_path == str(canonical_candidate)


@pytest.mark.parametrize("canonical_state", ["missing", "corrupt"])
def test_existing_published_run_independently_verifies_canonical_candidate_bytes(
    tmp_path: Path,
    canonical_state: str,
):
    export = _retry_export(tmp_path)
    evidence = _retry_evidence(tmp_path)
    canonical_candidate = tmp_path / "canonical-candidate.joblib"
    canonical_candidate.write_bytes(Path(export.candidate_artifact_path).read_bytes())
    evidence["row"]["candidate_artifact_path"] = str(canonical_candidate)
    if canonical_state == "missing":
        canonical_candidate.unlink()
    else:
        canonical_candidate.write_bytes(canonical_candidate.read_bytes() + b"corrupt")

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="existing candidate artifact failed verification",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            rate_package_id=42,
            allowed_artifact_root=tmp_path,
        )


def test_existing_published_run_returns_canonical_sql_ids_for_equivalent_retry(
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    evidence = _retry_evidence(tmp_path)
    canonical_artifact = save_candidate_bundle(
        CandidateBundle(
            fitted_model=SimpleNamespace(name="candidate"),
            X=pd.DataFrame({"age": [42.0]}),
            y=np.zeros(1),
            sample_weight=None,
            offset=None,
            export_weight=None,
            cv_report={},
            model_name="MTPL_FREQ",
            model_version="v1",
            export_id="canonical-export",
            manifest_id="canonical-manifest",
            split_set_id="canonical-split",
            pk_columns=("policy_id",),
            row_order_sha256=export.row_order_sha256,
            model_source_sha256=export.model_source_sha256,
            model_frame_sha256=export.model_frame_sha256,
            build_fingerprint_sha256=export.build_fingerprint_sha256,
            builder_source_sha256=export.builder_source_sha256,
            materialized_split_sha256=export.materialized_split_sha256,
            runtime_sha256=export.runtime_sha256,
            candidate_superglm_sha256=export.candidate_superglm_sha256,
            offset_contract={"handling": "NONE"},
        ),
        tmp_path / "canonical-candidate.joblib",
    )
    canonical_manifest = deepcopy(evidence["manifests"]["manifest-1"])
    canonical_manifest["manifest_id"] = "canonical-manifest"
    evidence["manifests"]["canonical-manifest"] = canonical_manifest
    evidence["columns"]["canonical-manifest"] = deepcopy(evidence["columns"]["manifest-1"])
    canonical_split = deepcopy(evidence["split_sets"]["split-1"])
    canonical_split.update(
        split_set_id="canonical-split",
        manifest_id="canonical-manifest",
    )
    evidence["split_sets"]["canonical-split"] = canonical_split
    evidence["split_geometry"]["canonical-split"] = deepcopy(evidence["split_geometry"]["split-1"])
    evidence["row"].update(
        source_export_id="canonical-export",
        run_export_id="canonical-export",
        manifest_id="canonical-manifest",
        candidate_artifact_path=str(canonical_artifact.path),
        candidate_artifact_sha256=canonical_artifact.sha256,
        candidate_artifact_size_bytes=canonical_artifact.size_bytes,
    )
    evidence["datasets"][0]["manifest_id"] = "canonical-manifest"
    evidence["splits"][0].update(
        manifest_id="canonical-manifest",
        split_set_id="canonical-split",
    )
    evidence["folds"][0]["split_set_id"] = "canonical-split"

    result = pipeline._resolve_existing_published_run(
        _Engine(_EvidenceConnection(evidence)),
        export,
        rate_package_id=42,
        allowed_artifact_root=tmp_path,
    )

    assert result.export_id == "canonical-export"
    assert result.manifest_id == "canonical-manifest"
    assert result.split_set_id == "canonical-split"
    assert result.candidate_artifact_path == str(canonical_artifact.path)


def test_existing_published_run_compares_complete_curve_points_exactly(tmp_path: Path):
    base = _retry_export(tmp_path).model_dump()
    point = {
        "validation_split_no": 1,
        "term_name": "age",
        "point_no": 1,
        "point_kind": "NUMERIC",
        "x_numeric": 0.0,
        "level_text": None,
        "eta_contribution": 0.0,
        "relativity": 1.0,
        "support_value": 1.0,
        "reference_value": 0.0,
        "reference_level": None,
    }
    base.update(
        validation_curve_status="COMPLETE",
        validation_curve_reason=None,
        validation_curve_points=(point,),
    )
    export = ModelExportResult(**base)
    evidence = _retry_evidence(tmp_path)
    evidence["row"].update(
        validation_curve_status="COMPLETE",
        validation_curve_reason=None,
    )
    evidence["curve_points"] = [
        {
            "split_set_id": "split-1",
            "split_no": 1,
            **{k: v for k, v in point.items() if k != "validation_split_no"},
        }
    ]

    pipeline._resolve_existing_published_run(
        _Engine(_EvidenceConnection(evidence)),
        export,
        allowed_artifact_root=tmp_path,
    )
    evidence["curve_points"][0]["eta_contribution"] = 0.25

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="validation curve points",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


def test_existing_published_run_rejects_fold_metric_owned_by_another_split(
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    evidence = _retry_evidence(tmp_path)
    evidence["folds"].append({**evidence["folds"][0], "split_set_id": "other-split"})

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="split metrics reference a non-canonical split_set_id",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


def test_existing_published_run_rejects_curve_point_owned_by_another_split(
    tmp_path: Path,
):
    base = _retry_export(tmp_path).model_dump()
    point = {
        "validation_split_no": 1,
        "term_name": "age",
        "point_no": 1,
        "point_kind": "NUMERIC",
        "x_numeric": 0.0,
        "level_text": None,
        "eta_contribution": 0.0,
        "relativity": 1.0,
        "support_value": 1.0,
        "reference_value": 0.0,
        "reference_level": None,
    }
    base.update(
        validation_curve_status="COMPLETE",
        validation_curve_reason=None,
        validation_curve_points=(point,),
    )
    export = ModelExportResult(**base)
    evidence = _retry_evidence(tmp_path)
    evidence["row"].update(
        validation_curve_status="COMPLETE",
        validation_curve_reason=None,
    )
    canonical_point = {
        "split_set_id": "split-1",
        "split_no": 1,
        **{k: v for k, v in point.items() if k != "validation_split_no"},
    }
    evidence["curve_points"] = [
        canonical_point,
        {**canonical_point, "split_set_id": "other-split"},
    ]

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="validation curve points reference a non-canonical split_set_id",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "field_name"),
    [
        (lambda evidence: evidence["row"].update(runtime_sha256="9" * 64), "runtime_sha256"),
        (_change_canonical_manifest_contract, "manifest contract"),
        (_change_canonical_manifest_columns, "manifest columns"),
        (_change_canonical_split_geometry, "split geometry"),
        (
            lambda evidence: evidence["metrics"][0].update(metric_value=9.0),
            "metrics",
        ),
        (
            lambda evidence: evidence["row"].update(validation_curve_reason="different reason"),
            "validation_curve",
        ),
    ],
)
def test_existing_published_run_rejects_conflicting_evidence(
    tmp_path: Path,
    mutation,
    field_name,
):
    evidence = deepcopy(_retry_evidence(tmp_path))
    mutation(evidence)

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match=rf"incompatible evidence.*{field_name}",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            _retry_export(tmp_path),
            allowed_artifact_root=tmp_path,
        )


def test_existing_published_run_verifies_candidate_bundle_identity(tmp_path: Path):
    evidence = _retry_evidence(tmp_path)
    metadata = save_candidate_bundle(
        CandidateBundle(
            fitted_model=SimpleNamespace(name="candidate"),
            X=pd.DataFrame({"age": [42.0]}),
            y=np.zeros(1),
            sample_weight=None,
            offset=None,
            export_weight=None,
            cv_report={},
            model_name="MTPL_FREQ",
            model_version="v1",
            export_id="export-1",
            manifest_id="manifest-1",
            split_set_id="split-1",
            pk_columns=("policy_id",),
            row_order_sha256="d" * 64,
            model_source_sha256="b" * 64,
            model_frame_sha256="e" * 64,
            build_fingerprint_sha256="1" * 64,
            builder_source_sha256="2" * 64,
            materialized_split_sha256="3" * 64,
            runtime_sha256="4" * 64,
            candidate_superglm_sha256="5" * 64,
            offset_contract={"handling": "NONE"},
        ),
        tmp_path / "candidate.joblib",
    )
    artifact_fields = {
        "candidate_artifact_path": metadata.path,
        "candidate_artifact_sha256": metadata.sha256,
        "candidate_artifact_format": metadata.format,
        "candidate_artifact_size_bytes": metadata.size_bytes,
        "candidate_python_version": metadata.python_version,
        "candidate_superglm_version": metadata.superglm_version,
        "candidate_superglm_git_sha": metadata.superglm_git_sha,
        "build_fingerprint_sha256": "1" * 64,
        "builder_source_sha256": "2" * 64,
        "materialized_split_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "candidate_superglm_sha256": "5" * 64,
        "row_order_sha256": "d" * 64,
        "model_source_sha256": "c" * 64,
    }
    evidence["row"].update(artifact_fields)
    export = ModelExportResult(**{**_retry_export(tmp_path).model_dump(), **artifact_fields})

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="candidate artifact model_source_sha256 does not match model-run lineage",
    ):
        pipeline._resolve_existing_published_run(
            _Engine(_EvidenceConnection(evidence)),
            export,
            allowed_artifact_root=tmp_path,
        )


def test_publish_model_export_stages_packages_and_records_lineage(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    export = _retry_export(tmp_path)
    connection = object()

    monkeypatch.setattr(
        pipeline,
        "stage_rating_export",
        lambda engine, **kwargs: calls.append(("stage", engine, kwargs)) or "a" * 64,
    )

    def publish(engine, **kwargs):
        calls.append(("publish", engine, kwargs))
        model_run_id = kwargs["package_lineage_writer"](connection, 42)
        return PublishResult(
            mlflow_run_id="",
            export_id="export-1",
            rate_package_id=42,
            package_version=3,
            rating_workbook_path="",
            package_status="PUBLISHED",
            model_run_id=model_run_id,
        )

    monkeypatch.setattr(pipeline, "publish_rating_package", publish)
    monkeypatch.setattr(
        pipeline,
        "record_model_run",
        lambda engine, *, connection, **kwargs: (
            calls.append(("lineage", connection, kwargs)) or 501
        ),
    )

    result = pipeline.publish_model_export(
        object(),
        export,
        model_config=MODEL_CONFIG,
        expected_database="PricingLab",
        allowed_artifact_root=tmp_path,
        validated_model_id=17,
    )

    assert isinstance(result, CompletedModelPublishResult)
    assert result.rate_package_id == 42
    assert result.model_run_id == 501
    assert result.was_existing is False
    stage_call = next(call for call in calls if call[0] == "stage")
    assert stage_call[2]["workbook_path"] == Path(export.rating_workbook_path)
    assert stage_call[2]["publication_receipt_path"] == Path(export.publication_receipt_path)
    publish_call = next(call for call in calls if call[0] == "publish")
    assert "package_status" not in publish_call[2]
    assert publish_call[2]["expected_staged_metadata"]["staging_content_sha256"] == "a" * 64
    lineage_call = next(call for call in calls if call[0] == "lineage")
    assert lineage_call[1] is connection
    assert lineage_call[2]["build"] is export
    assert lineage_call[2]["rate_package_id"] == 42


@pytest.mark.parametrize(
    ("artifact_state", "match"),
    [
        ("outside_workbook", "rating workbook is outside"),
        ("outside_receipt", "publication receipt is outside"),
        ("missing_receipt", "publication receipt does not exist"),
        ("tampered_receipt", "publication receipt SHA-256"),
    ],
)
def test_publish_model_export_preflights_root_artifacts_before_staging(
    monkeypatch,
    tmp_path: Path,
    artifact_state: str,
    match: str,
):
    artifact_root = tmp_path / "workbench"
    artifact_root.mkdir()
    export = _retry_export(artifact_root)
    if artifact_state == "outside_workbook":
        outside_workbook = tmp_path / "outside-rating.xlsx"
        outside_workbook.write_bytes(Path(export.rating_workbook_path).read_bytes())
        export = export.model_copy(update={"rating_workbook_path": str(outside_workbook)})
    elif artifact_state == "outside_receipt":
        outside_receipt = tmp_path / "outside-receipt.json"
        outside_receipt.write_bytes(Path(export.publication_receipt_path).read_bytes())
        export = export.model_copy(update={"publication_receipt_path": str(outside_receipt)})
    elif artifact_state == "missing_receipt":
        Path(export.publication_receipt_path).unlink()
    else:
        receipt = Path(export.publication_receipt_path)
        receipt.write_bytes(receipt.read_bytes() + b"tampered")

    monkeypatch.setattr(
        pipeline,
        "stage_rating_export",
        lambda *args, **kwargs: pytest.fail("untrusted artifact reached staging"),
    )
    monkeypatch.setattr(
        pipeline,
        "publish_rating_package",
        lambda *args, **kwargs: pytest.fail("untrusted artifact reached packaging"),
    )

    with pytest.raises(pipeline.PublishedRunIntegrityError, match=match):
        pipeline.publish_model_export(
            object(),
            export,
            model_config=MODEL_CONFIG,
            expected_database="PricingLab",
            validated_model_id=17,
            allowed_artifact_root=artifact_root,
        )


def test_publish_model_export_requires_artifact_root_before_staging(
    monkeypatch,
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "stage_rating_export",
        lambda *args, **kwargs: pytest.fail("missing root reached staging"),
    )

    with pytest.raises(
        pipeline.PublishedRunIntegrityError,
        match="allowed_artifact_root is required",
    ):
        pipeline.publish_model_export(
            object(),
            export,
            model_config=MODEL_CONFIG,
            expected_database="PricingLab",
            allowed_artifact_root=None,
            validated_model_id=17,
        )


def test_publish_model_export_returns_verified_existing_result_without_rewriting_lineage(
    monkeypatch,
    tmp_path: Path,
):
    export = _retry_export(tmp_path)
    existing = CompletedModelPublishResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="v1",
        manifest_id="manifest-1",
        split_set_id="split-1",
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        package_status="PUBLISHED",
        rating_workbook_path=export.rating_workbook_path,
        model_run_id=501,
        was_existing=True,
    )
    monkeypatch.setattr(pipeline, "stage_rating_export", lambda *args, **kwargs: "a" * 64)
    monkeypatch.setattr(
        pipeline,
        "publish_rating_package",
        lambda *args, **kwargs: PublishResult(
            mlflow_run_id="",
            export_id="export-1",
            rate_package_id=42,
            package_version=3,
            rating_workbook_path="",
            package_status="PUBLISHED",
            was_existing=True,
        ),
    )
    monkeypatch.setattr(
        pipeline, "_resolve_existing_published_run", lambda *args, **kwargs: existing
    )
    monkeypatch.setattr(
        pipeline,
        "record_model_run",
        lambda *args, **kwargs: pytest.fail("existing publication must not rewrite lineage"),
    )

    result = pipeline.publish_model_export(
        object(),
        export,
        model_config=MODEL_CONFIG,
        expected_database="PricingLab",
        allowed_artifact_root=tmp_path,
        validated_model_id=17,
    )

    assert result is existing
