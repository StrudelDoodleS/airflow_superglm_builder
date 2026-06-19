from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from pricing_pipeline.orchestration import pipeline
from pricing_pipeline.publishing import lineage, rating_export, rating_package, staging
from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
    write_publication_receipt,
)
from pricing_pipeline.infra.config import Settings
from pricing_pipeline.data.datasets import FREMTPL_DATASET_SPEC
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import ModelSpec
from pricing_pipeline.models.spec import ModelExportResult
from pricing_pipeline.publishing.publisher import ModelPublisher
from pricing_pipeline.publishing.model_registry import ModelRegistryError, ensure_pricing_model
from pricing_models.mtpl_frequency.training import (
    FEATURE_COLUMNS,
    TRAINING_SQL,
    build_training_frame,
)
from pricing_models.mtpl_frequency.spec import MODEL_CONFIG
from scripts import load_staging_to_rating_package
from scripts import load_superglm_excel_to_staging
from scripts import smoke_check


def raw_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [0, 1],
            "Exposure": [0.5, 2.0],
            "Area": ["A", "B"],
            "VehPower": [6, 7],
            "VehAge": [3, 4],
            "DrivAge": [45, 50],
            "BonusMalus": [50, 60],
            "VehBrand": ["B1", "B2"],
            "VehGas": ["Regular", "Diesel"],
            "Density": [1.0, 123.0],
            "Region": ["R1", "R2"],
        }
    )


def test_build_export_id_is_path_safe():
    export_id = rating_export.build_export_id("MTPL_FREQ", "scheduled__2026-04-27T10:30:00+00:00")

    assert export_id == "mtpl_freq__scheduled__20260427t1030000000"


def test_build_rating_export_path_uses_model_and_date(tmp_path: Path):
    path = rating_export.build_rating_export_path(
        tmp_path,
        model_name="MTPL_FREQ",
        logical_date="2026-04-27",
        export_id="mtpl_freq__run1",
    )

    assert path == tmp_path / "MTPL_FREQ" / "2026-04-27" / "mtpl_freq__run1" / "rating_tables.xlsx"


def test_build_rating_export_path_accepts_positional_call_shape(tmp_path: Path):
    path = rating_export.build_rating_export_path(
        tmp_path, "MTPL_FREQ", "2026-04-27", "mtpl_freq__run1"
    )

    assert path == tmp_path / "MTPL_FREQ" / "2026-04-27" / "mtpl_freq__run1" / "rating_tables.xlsx"


class FakeExportModel:
    def __init__(self):
        self.calls = []

    def export_rating_tables(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        Path(args[0]).write_bytes(b"workbook")


def test_export_rating_tables_creates_parent_calls_model_and_logs_mlflow(
    monkeypatch, tmp_path: Path
):
    model = FakeExportModel()
    mlflow_calls = []
    fake_mlflow = SimpleNamespace(
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append((path, artifact_path))
    )
    monkeypatch.setattr(rating_export, "mlflow", fake_mlflow)

    output_path = tmp_path / "nested" / "rating_tables.xlsx"
    X = pd.DataFrame({"x": [1, 2]})
    y = np.array([0.0, 1.0])
    exposure = np.array([0.5, 2.0])

    result = rating_export.export_rating_tables(model, X, y, exposure, output_path=output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"
    assert model.calls == [((output_path, X, y), {"sample_weight": exposure, "n_bins": 150})]
    assert mlflow_calls == [(str(output_path), "rating_tables")]


def test_export_rating_tables_accepts_positional_output_path(monkeypatch, tmp_path: Path):
    model = FakeExportModel()
    mlflow_calls = []
    fake_mlflow = SimpleNamespace(
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append((path, artifact_path))
    )
    monkeypatch.setattr(rating_export, "mlflow", fake_mlflow)

    output_path = tmp_path / "nested" / "rating_tables.xlsx"
    X = pd.DataFrame({"x": [1, 2]})
    y = np.array([0.0, 1.0])
    exposure = np.array([0.5, 2.0])

    result = rating_export.export_rating_tables(model, X, y, exposure, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"
    assert model.calls == [((output_path, X, y), {"sample_weight": exposure, "n_bins": 150})]
    assert mlflow_calls == [(str(output_path), "rating_tables")]


def test_export_rating_tables_requires_superglm_rating_export_support(monkeypatch, tmp_path: Path):
    mlflow_calls = []
    fake_mlflow = SimpleNamespace(
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append((path, artifact_path))
    )
    monkeypatch.setattr(rating_export, "mlflow", fake_mlflow)

    output_path = tmp_path / "nested" / "rating_tables.xlsx"

    with pytest.raises(RuntimeError) as exc:
        rating_export.export_rating_tables(
            object(),
            pd.DataFrame({"x": [1]}),
            np.array([0.0]),
            np.array([1.0]),
            output_path,
        )

    message = str(exc.value)
    assert "SuperGLM" in message
    assert "export_rating_tables" in message
    assert "PR #109" in message
    assert "rating table export support" in message
    assert not output_path.parent.exists()
    assert mlflow_calls == []


def test_export_rating_tables_ignores_mlflow_logging_failure(monkeypatch, tmp_path: Path):
    model = FakeExportModel()

    class FailingMlflow:
        def log_artifact(self, path, artifact_path=None):
            raise RuntimeError("mlflow unavailable")

    monkeypatch.setattr(rating_export, "mlflow", FailingMlflow())

    output_path = tmp_path / "rating_tables.xlsx"
    result = rating_export.export_rating_tables(
        model,
        pd.DataFrame({"x": [1]}),
        np.array([0.0]),
        np.array([1.0]),
        output_path,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"


def test_export_rating_tables_can_use_disabled_mlflow_client(monkeypatch, tmp_path: Path):
    model = FakeExportModel()

    class RaisingMlflow:
        def log_artifact(self, path, artifact_path=None):
            raise AssertionError("global mlflow should not be used")

    monkeypatch.setattr(rating_export, "mlflow", RaisingMlflow())

    output_path = tmp_path / "rating_tables.xlsx"
    result = rating_export.export_rating_tables(
        model,
        pd.DataFrame({"x": [1]}),
        np.array([0.0]),
        np.array([1.0]),
        output_path,
        mlflow_client=None,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"


def test_smoke_check_reports_missing_rating_export_without_failing(capsys):
    class OldSuperGLM:
        pass

    exit_code = smoke_check.check_superglm_rating_export(OldSuperGLM)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "smoke_check=rating_export_unavailable" in captured.out
    assert "PR #109" in captured.out
    assert "export_rating_tables" in captured.out


class FakeFrame:
    def __init__(self, label: str, events: list[tuple], *, empty: bool = False):
        self.label = label
        self.events = events
        self.empty = empty

    def to_sql(self, name, con, schema=None, if_exists=None, index=None, chunksize=None):
        self.events.append(("to_sql", self.label, name, con, schema, if_exists, index, chunksize))


class FakeBeginConnection:
    def __init__(self, events: list[tuple]):
        self.events = events

    def execute(self, statement, params=None):
        self.events.append(("execute", str(statement), params))


class FakeBegin:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self, events: list[tuple], execution_options=None):
        self.events = events
        self._execution_options = execution_options or {}
        self.connection = FakeBeginConnection(events)

    def begin(self):
        self.events.append(("begin",))
        return FakeBegin(self.connection)


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class FakePricingModelResult:
    def mappings(self):
        return self

    def one_or_none(self):
        return {
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_label": None,
            "target_name": "ClaimNb",
            "model_type": "superglm_poisson",
            "model_status": "ACTIVE",
        }


class FakeModelRegistryConnection(FakeBeginConnection):
    def execute(self, statement, params=None):
        self.events.append(("execute", str(statement), params))
        if "SELECT model_id," in str(statement):
            return FakePricingModelResult()
        if "SELECT model_id" in str(statement):
            return FakeScalarResult(17)
        return FakeScalarResult(None)


class FakeModelRegistryEngine(FakeEngine):
    def __init__(self, events: list[tuple], execution_options=None):
        self.events = events
        self._execution_options = execution_options or {}
        self.connection = FakeModelRegistryConnection(events)


def test_ensure_pricing_model_merges_by_model_name_and_returns_model_id():
    events = []
    con = FakeModelRegistryConnection(events)

    model_id = ensure_pricing_model(
        con,
        model_name="MTPL_FREQ",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        created_by="airflow",
    )

    assert model_id == 17
    merge_sql = events[0][1]
    select_sql = events[1][1]
    assert "MERGE pricing.PRICING_MODEL WITH (HOLDLOCK)" in merge_sql
    assert "ON tgt.model_name = src.model_name" in merge_sql
    assert "model_status" in merge_sql
    assert "SELECT model_id" in select_sql
    assert events[0][2] == {
        "model_name": "MTPL_FREQ",
        "model_label": None,
        "target_name": "ClaimNb",
        "model_type": "superglm_poisson",
        "model_status": "ACTIVE",
        "created_by": "airflow",
    }


def test_package_staging_module_exposes_excel_staging_functions():
    from pricing_pipeline.publishing import staging

    assert callable(staging.build_staging_frames)
    assert callable(staging.insert_staging_frames)
    assert callable(staging.stage_rating_export)


def _offline_staging_engine(tmp_path: Path):
    from scripts.run_mtpl_frequency_offline_sqlite import (
        apply_offline_ddl,
        sqlite_engine_with_offline_schemas,
    )

    engine = sqlite_engine_with_offline_schemas(
        {
            "pricing": tmp_path / "pricing.sqlite",
            "pricing_stg": tmp_path / "pricing_stg.sqlite",
            "mlops": tmp_path / "mlops.sqlite",
        }
    )
    apply_offline_ddl(engine)
    return engine


def _minimal_rating_workbook(path: Path, term_name: str = "TermMonths") -> Path:
    raw = pd.DataFrame([[None] * 3 for _ in range(8)])
    raw.iat[1, 2] = 0.123
    raw.iat[4, 0] = term_name
    raw.iloc[6, 0:3] = [term_name, "Relativity", "Weight"]
    raw.iloc[7, 0:3] = ["12", 1.0, 10.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_excel(path, sheet_name="Rating Tables", header=False, index=False)
    return path


def _publication_receipt(
    *,
    handling: str = "EXPORTED_FACTOR",
    published_factor_name: str | None = "TermMonths",
) -> SuperGLMPublicationReceipt:
    if handling == "EXPORTED_FACTOR":
        offset_contract = OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="term_months",
            published_factor_name=published_factor_name,
            source_name="Exposure",
            label="Policy exposure term",
        )
        term_metadata = {
            "TermMonths": {
                "feature_kind": "offset",
                "source_column": "Exposure",
            }
        }
    elif handling == "ALREADY_APPLIED_SQL_EXPOSURE":
        offset_contract = OffsetExportContract(
            handling="ALREADY_APPLIED_SQL_EXPOSURE",
            source_name="Exposure",
            label="Policy exposure term",
        )
        term_metadata = {}
    else:
        offset_contract = OffsetExportContract(handling="NONE")
        term_metadata = {}

    return SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version="1.0.0",
        extractor_version="unit-test",
        package_metadata={"model": {"family": "poisson", "link": "log"}},
        term_metadata=term_metadata,
        offset_contract=offset_contract,
    )


def test_stage_rating_export_stages_publication_receipt_and_offset_metadata(
    tmp_path: Path,
):
    engine = _offline_staging_engine(tmp_path)
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")
    receipt_path = tmp_path / "superglm_publication_receipt.json"
    digest = write_publication_receipt(_publication_receipt(), receipt_path)

    load_superglm_excel_to_staging.stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from="2026-04-27",
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=digest,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
        replace=True,
        model_id=1,
    )

    with engine.begin() as con:
        export = con.execute(
            text(
                "SELECT publication_receipt_sha256, offset_handling, offset_factor_name "
                "FROM pricing_stg.STG_RATING_EXPORT WHERE export_id = :export_id"
            ),
            {"export_id": "export-1"},
        ).mappings().one()
        term = con.execute(
            text(
                "SELECT term_type FROM pricing_stg.STG_RATE_CELL "
                "WHERE export_id = :export_id AND term_name = :term_name"
            ),
            {"export_id": "export-1", "term_name": "TermMonths"},
        ).mappings().one()
        metadata = con.execute(
            text(
                "SELECT term_metadata_json FROM pricing_stg.STG_TERM_METADATA "
                "WHERE export_id = :export_id AND term_name = :term_name"
            ),
            {"export_id": "export-1", "term_name": "TermMonths"},
        ).mappings().one()

    assert export["publication_receipt_sha256"] == digest
    assert export["offset_handling"] == "EXPORTED_FACTOR"
    assert export["offset_factor_name"] == "TermMonths"
    assert term["term_type"] == "OFFSET_FACTOR"
    assert json.loads(metadata["term_metadata_json"])["feature_kind"] == "offset"


def test_stage_rating_export_requires_publication_receipt_in_strict_metadata_mode(
    tmp_path: Path,
):
    engine = _offline_staging_engine(tmp_path)
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")

    with pytest.raises(ValueError, match="publication receipt is required"):
        load_superglm_excel_to_staging.stage_rating_export(
            engine,
            workbook_path=workbook_path,
            export_id="export-1",
            model_name="MTPL_FREQ",
            model_version="20260427",
            effective_from="2026-04-27",
            metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
            model_id=1,
        )


def test_stage_rating_export_allows_explicit_workbook_only_metadata_mode(
    tmp_path: Path,
):
    engine = _offline_staging_engine(tmp_path)
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")

    load_superglm_excel_to_staging.stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from="2026-04-27",
        metadata_mode="ALLOW_WORKBOOK_ONLY",
        replace=True,
        model_id=1,
    )

    with engine.begin() as con:
        export = con.execute(
            text(
                "SELECT publication_receipt_sha256, offset_handling "
                "FROM pricing_stg.STG_RATING_EXPORT WHERE export_id = :export_id"
            ),
            {"export_id": "export-1"},
        ).mappings().one()
        term_metadata_count = con.execute(
            text(
                "SELECT COUNT(*) FROM pricing_stg.STG_TERM_METADATA "
                "WHERE export_id = :export_id"
            ),
            {"export_id": "export-1"},
        ).scalar_one()

    assert export["publication_receipt_sha256"] is None
    assert export["offset_handling"] is None
    assert term_metadata_count == 0


def test_stage_rating_export_rejects_missing_exported_offset_factor(
    tmp_path: Path,
):
    engine = _offline_staging_engine(tmp_path)
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")
    receipt_path = tmp_path / "superglm_publication_receipt.json"
    digest = write_publication_receipt(
        _publication_receipt(published_factor_name="MissingOffsetFactor"),
        receipt_path,
    )

    with pytest.raises(ValueError, match="MissingOffsetFactor"):
        load_superglm_excel_to_staging.stage_rating_export(
            engine,
            workbook_path=workbook_path,
            export_id="export-1",
            model_name="MTPL_FREQ",
            model_version="20260427",
            effective_from="2026-04-27",
            publication_receipt_path=receipt_path,
            publication_receipt_sha256=digest,
            metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
            replace=True,
            model_id=1,
        )


def test_stage_rating_export_replace_deletes_old_term_metadata(
    tmp_path: Path,
):
    engine = _offline_staging_engine(tmp_path)
    workbook_path = _minimal_rating_workbook(tmp_path / "rating_tables.xlsx")
    receipt_path = tmp_path / "superglm_publication_receipt.json"
    digest = write_publication_receipt(_publication_receipt(), receipt_path)

    load_superglm_excel_to_staging.stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from="2026-04-27",
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=digest,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
        replace=True,
        model_id=1,
    )

    with engine.begin() as con:
        con.execute(
            text(
                "INSERT INTO pricing_stg.STG_TERM_METADATA "
                "(export_id, term_name, term_metadata_json) "
                "VALUES (:export_id, :term_name, :term_metadata_json)"
            ),
            {
                "export_id": "export-1",
                "term_name": "StaleTerm",
                "term_metadata_json": '{"feature_kind":"stale"}',
            },
        )

    load_superglm_excel_to_staging.stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from="2026-04-27",
        publication_receipt_path=receipt_path,
        publication_receipt_sha256=digest,
        metadata_mode="REQUIRE_SUPERGLM_RECEIPT",
        replace=True,
        model_id=1,
    )

    with engine.begin() as con:
        term_names = con.execute(
            text(
                "SELECT term_name FROM pricing_stg.STG_TERM_METADATA "
                "WHERE export_id = :export_id ORDER BY term_name"
            ),
            {"export_id": "export-1"},
        ).scalars().all()

    assert term_names == ["TermMonths"]


def test_stage_rating_export_builds_args_deletes_and_inserts_in_one_transaction(
    monkeypatch, tmp_path: Path
):
    events = []
    captured_args = []

    def fake_build_staging_frames(args):
        captured_args.append(args)
        return (
            FakeFrame("export", events),
            FakeFrame("rate", events),
            FakeFrame("level", events),
        )

    monkeypatch.setattr(staging, "build_staging_frames", fake_build_staging_frames)
    engine = FakeModelRegistryEngine(events)
    workbook_path = tmp_path / "rating_tables.xlsx"

    load_superglm_excel_to_staging.stage_rating_export(
        engine,
        workbook_path=workbook_path,
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from="2026-04-27",
        effective_to=None,
        created_by="airflow",
        replace=True,
        metadata_mode="ALLOW_WORKBOOK_ONLY",
    )

    args = captured_args[0]
    assert args.xlsx == workbook_path
    assert args.sheet == "Rating Tables"
    assert args.export_id == "export-1"
    assert args.model_name == "MTPL_FREQ"
    assert args.target_name == "ClaimNb"
    assert args.model_type == "superglm_poisson"
    assert args.model_status == "ACTIVE"
    assert args.model_version == "20260427"
    assert args.effective_from == "2026-04-27"
    assert args.effective_to is None
    assert args.created_by == "airflow"
    assert args.replace is True

    delete_sql = [event[1] for event in events if event[0] == "execute"]
    assert delete_sql == [
        "\n                SELECT model_id,\n                    model_name,\n                    model_label,\n                    target_name,\n                    model_type,\n                    model_status\n                FROM pricing.PRICING_MODEL\n                WHERE model_name = :model_name\n                ",
        "DELETE FROM pricing_stg.STG_TERM_METADATA WHERE export_id = :export_id",
        "DELETE FROM pricing_stg.STG_CELL_LEVEL WHERE export_id = :export_id",
        "DELETE FROM pricing_stg.STG_RATE_CELL WHERE export_id = :export_id",
        "DELETE FROM pricing_stg.STG_RATING_EXPORT WHERE export_id = :export_id",
        "UPDATE pricing_stg.STG_RATING_EXPORT SET model_id = :model_id WHERE export_id = :export_id",
    ]
    assert [event[2] for event in events if event[0] == "execute"] == [
        {"model_name": "MTPL_FREQ"},
        {"export_id": "export-1"},
        {"export_id": "export-1"},
        {"export_id": "export-1"},
        {"export_id": "export-1"},
        {"export_id": "export-1", "model_id": 17},
    ]
    assert [event[:4] for event in events if event[0] == "to_sql"] == [
        ("to_sql", "export", "STG_RATING_EXPORT", engine.connection),
        ("to_sql", "rate", "STG_RATE_CELL", engine.connection),
        ("to_sql", "level", "STG_CELL_LEVEL", engine.connection),
    ]
    assert [event[-1] for event in events if event[0] == "to_sql"] == [
        None,
        5000,
        5000,
    ]
    assert events[0] == ("begin",)


def test_insert_staging_frames_uses_configured_staging_schema_for_to_sql():
    events = []
    engine = FakeModelRegistryEngine(
        events,
        {"pricing_staging_schema": "python_pricing_stg"},
    )
    args = SimpleNamespace(
        model_name="MTPL_FREQ",
        model_label=None,
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_status="ACTIVE",
        created_by="unit-test",
        replace=False,
        export_id="export-1",
    )

    staging.insert_staging_frames(
        engine,
        args,
        FakeFrame("export", events),
        FakeFrame("rate", events),
        FakeFrame("level", events),
    )

    assert [event[4] for event in events if event[0] == "to_sql"] == [
        "python_pricing_stg",
        "python_pricing_stg",
        "python_pricing_stg",
    ]


def test_insert_staging_frames_writes_term_metadata_after_cell_levels():
    events = []
    engine = FakeEngine(events)
    args = SimpleNamespace(
        model_name="MTPL_FREQ",
        model_label=None,
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_status="ACTIVE",
        created_by="unit-test",
        replace=False,
        export_id="export-1",
        model_id=17,
    )

    staging.insert_staging_frames(
        engine,
        args,
        FakeFrame("export", events),
        FakeFrame("rate", events),
        FakeFrame("level", events),
        FakeFrame("term_metadata", events),
    )

    assert [event[1] for event in events if event[0] == "to_sql"] == [
        "export",
        "rate",
        "level",
        "term_metadata",
    ]
    assert [event[2] for event in events if event[0] == "to_sql"] == [
        "STG_RATING_EXPORT",
        "STG_RATE_CELL",
        "STG_CELL_LEVEL",
        "STG_TERM_METADATA",
    ]


class FakeMissingModelConnection(FakeBeginConnection):
    def execute(self, statement, params=None):
        self.events.append(("execute", str(statement), params))
        if "FROM pricing.PRICING_MODEL" in str(statement):
            return SimpleNamespace(mappings=lambda: SimpleNamespace(one_or_none=lambda: None))
        raise AssertionError("staging should stop before writing when model is missing")


class FakeMissingModelEngine(FakeEngine):
    def __init__(self, events: list[tuple]):
        self.events = events
        self._execution_options = {}
        self.connection = FakeMissingModelConnection(events)


def test_insert_staging_frames_requires_existing_registered_model(monkeypatch):
    events = []
    engine = FakeMissingModelEngine(events)
    args = SimpleNamespace(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        model_status="ACTIVE",
        created_by="unit-test",
        replace=False,
        export_id="export-1",
    )
    monkeypatch.setattr(
        staging,
        "ensure_pricing_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("staging must not create model registry rows")
        ),
        raising=False,
    )

    with pytest.raises(ModelRegistryError, match="not registered"):
        staging.insert_staging_frames(
            engine,
            args,
            FakeFrame("export", events),
            FakeFrame("rate", events),
            FakeFrame("level", events),
        )


def test_build_staging_frames_accepts_superglm_export_headers(monkeypatch, tmp_path: Path):
    raw = pd.DataFrame([[None] * 6 for _ in range(10)])
    raw.iat[1, 2] = 0.123
    raw.iat[4, 0] = "VehAge"
    raw.iat[4, 3] = "DrivAge"
    raw.iloc[6, 0:3] = ["VehAge", "Relativity", "Weight"]
    raw.iloc[6, 3:6] = ["DrivAge", "Relativity", "Weight"]
    raw.iloc[7, 0:3] = ["[0, 1)", 1.2, 10.0]
    raw.iloc[8, 0:3] = ["[1, 2)", 0.9, 20.0]
    raw.iloc[7, 3:6] = ["[18, 20)", 1.1, 30.0]

    monkeypatch.setattr(
        staging.pd,
        "read_excel",
        lambda *args, **kwargs: raw,
    )

    args = SimpleNamespace(
        xlsx=tmp_path / "rating_tables.xlsx",
        sheet="Rating Tables",
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        effective_from="2026-04-27",
        effective_to=None,
        base_rate=None,
        base_rate_cell="C2",
        term_row=5,
        header_row=7,
        data_start_row=8,
        term_type_map_json="{}",
        interaction_features_json="{}",
        created_by="airflow",
    )

    export_df, rate_df, level_df = load_superglm_excel_to_staging.build_staging_frames(args)

    assert export_df.loc[0, "base_rate"] == 0.123
    assert rate_df["term_name"].tolist() == ["VehAge", "VehAge", "DrivAge"]
    assert rate_df["multiplier"].tolist() == [1.2, 0.9, 1.1]
    assert level_df["feature_name"].tolist() == ["VehAge", "VehAge", "DrivAge"]


def test_publish_rating_package_wrapper_returns_publish_result_without_pointer(monkeypatch):
    captured = []

    def fake_load(engine, args):
        captured.append((engine, args))
        args.package_version = 3
        return 42

    monkeypatch.setattr(
        "pricing_pipeline.publishing.package_writer.load_staging_to_rating_package",
        fake_load,
    )
    engine = object()

    result = rating_package.publish_rating_package(
        engine,
        export_id="export-1",
        created_by="airflow",
        package_status="PUBLISHED",
    )

    assert result == PublishResult(
        mlflow_run_id="",
        export_id="export-1",
        rate_package_id=42,
        package_version=3,
        rating_workbook_path="",
    )
    assert captured[0][0] is engine
    args = captured[0][1]
    assert args.export_id == "export-1"
    assert args.created_by == "airflow"
    assert args.package_status == "PUBLISHED"
    assert args.set_pointer is None


def test_publish_rating_package_wrapper_rejects_legacy_pointer_api():
    engine = object()

    with pytest.raises(ValueError, match="deploy"):
        rating_package.publish_rating_package(
            engine,
            export_id="export-legacy",
            pointer_name="MTPL_FREQ_UAT",
            created_by="airflow",
            package_status="DRAFT",
        )


def test_publish_script_callable_rejects_legacy_pointer_api():
    engine = object()

    with pytest.raises(ValueError, match="deploy"):
        load_staging_to_rating_package.publish_rating_package(
            engine,
            export_id="export-2",
            pointer_name="MTPL_FREQ_UAT",
            created_by="python",
            package_status="DRAFT",
        )


def test_publish_script_callable_builds_args_and_returns_package_id(monkeypatch):
    captured = []

    def fake_load(engine, args):
        captured.append((engine, args))
        return 99

    monkeypatch.setattr(
        load_staging_to_rating_package,
        "load_staging_to_rating_package",
        fake_load,
    )
    engine = object()

    package_id = load_staging_to_rating_package.publish_rating_package(
        engine,
        export_id="export-2",
        created_by="python",
        package_status="DRAFT",
    )

    assert package_id == 99
    assert captured[0][0] is engine
    args = captured[0][1]
    assert args.export_id == "export-2"
    assert args.set_pointer is None
    assert args.created_by == "python"
    assert args.package_status == "DRAFT"


def test_model_publisher_publish_training_export_uses_config_and_maps_result(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    engine = object()
    workbook_path = tmp_path / "rating_tables.xlsx"
    config = ModelBuildConfig(
        model_name="CONFIG_MODEL",
        model_label="Config model",
        target_name="ConfigTarget",
        model_type="config_type",
        deployment_slot="CONFIG_UAT",
        default_package_status="PUBLISHED",
    )
    export = ModelExportResult(
        model_id=17,
        model_name="CONFIG_MODEL",
        model_version="20260527",
        model_type="config_type",
        target_name="ConfigTarget",
        deployment_slot="CONFIG_UAT",
        manifest_id="manifest-1",
        dag_id="dag",
        airflow_run_id="scheduled__2026-05-27",
        mlflow_run_id="mlflow-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(workbook_path),
        effective_from="2026-05-27",
        created_by="airflow",
        package_status="DRAFT",
    )

    def fake_stage_rating_export(engine_arg, **kwargs):
        calls.append(("stage", engine_arg, kwargs))

    def fake_publish_rating_package(engine_arg, **kwargs):
        calls.append(("publish", engine_arg, kwargs))
        return PublishResult(
            mlflow_run_id="",
            export_id="export-1",
            rate_package_id=42,
            package_version=6,
            rating_workbook_path="",
        )

    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.validate_model_on_engine",
        lambda engine_arg, config_arg: 17,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.stage_rating_export",
        fake_stage_rating_export,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.publish_rating_package",
        fake_publish_rating_package,
    )

    result = ModelPublisher(engine, config).publish_training_export(export)

    assert result == PublishResult(
        mlflow_run_id="mlflow-1",
        export_id="export-1",
        rate_package_id=42,
        package_version=6,
        rating_workbook_path=str(workbook_path),
    )
    assert calls == [
        (
            "stage",
            engine,
            {
                "workbook_path": workbook_path,
                "export_id": "export-1",
                "model_name": "CONFIG_MODEL",
                "model_version": "20260527",
                "target_name": "ConfigTarget",
                "model_type": "config_type",
                "effective_from": "2026-05-27",
                "created_by": "airflow",
                "replace": True,
                "model_id": 17,
            },
        ),
        (
            "publish",
            engine,
            {
                "export_id": "export-1",
                "created_by": "airflow",
                "package_status": "PUBLISHED",
            },
        ),
    ]


def test_record_model_run_uses_parameterized_sql_with_expected_params():
    events = []
    engine = FakeLineageEngine(events)

    lineage.record_model_run(
        engine,
        dag_id="dag",
        airflow_run_id="scheduled__2026-04-27",
        mlflow_run_id="mlflow-1",
        manifest_id="manifest-1",
        export_id="export-1",
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="20260427",
        split_set_id="manifest-1__kfold_5_seed_42",
        rate_package_id=42,
        rating_workbook_path="/tmp/rating_tables.xlsx",
        run_status="SUCCESS",
        created_by="airflow",
    )

    assert len(events) == 5
    sql = events[1][1]
    params = events[1][2]
    assert "MERGE pricing.MODEL_RUN" in sql
    assert "WHEN MATCHED THEN" in sql
    assert "WHEN NOT MATCHED THEN" in sql
    assert "tgt.dag_id = src.dag_id" in sql
    assert "tgt.airflow_run_id = src.airflow_run_id" in sql
    assert "tgt.model_id = src.model_id" in sql
    assert "SYSUTCDATETIME()" in sql
    assert ":dag_id" in sql
    assert params == {
        "dag_id": "dag",
        "airflow_run_id": "scheduled__2026-04-27",
        "mlflow_run_id": "mlflow-1",
        "manifest_id": "manifest-1",
        "export_id": "export-1",
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "20260427",
        "split_set_id": "manifest-1__kfold_5_seed_42",
        "dataset_role": "training",
        "split_role": "validation",
        "rate_package_id": 42,
        "rating_workbook_path": "/tmp/rating_tables.xlsx",
        "run_status": "SUCCESS",
        "created_by": "airflow",
    }


class FakeLineageConnection(FakeBeginConnection):
    def execute(self, statement, params=None):
        self.events.append(("execute", str(statement), params))
        if "SELECT model_run_id" in str(statement):
            return FakeScalarResult(501)
        return FakeScalarResult(None)


class FakeLineageEngine(FakeEngine):
    def __init__(self, events: list[tuple]):
        self.events = events
        self.connection = FakeLineageConnection(events)


def test_record_model_run_links_run_to_dataset_and_split_set():
    events = []
    engine = FakeLineageEngine(events)

    model_run_id = lineage.record_model_run(
        engine,
        dag_id="dag",
        airflow_run_id="scheduled__2026-04-27",
        mlflow_run_id="mlflow-1",
        manifest_id="manifest-1",
        split_set_id="manifest-1__kfold_5_seed_42",
        export_id="export-1",
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="20260427",
        rate_package_id=42,
        rating_workbook_path="/tmp/rating_tables.xlsx",
        run_status="SUCCESS",
        created_by="airflow",
    )

    assert model_run_id == 501
    executed_sql = [event[1] for event in events if event[0] == "execute"]
    assert "MERGE pricing.MODEL_RUN" in executed_sql[0]
    assert "SELECT model_run_id" in executed_sql[1]
    assert "MERGE mlops.MODEL_RUN_DATASET" in executed_sql[2]
    assert "MERGE mlops.MODEL_RUN_SPLIT_SET" in executed_sql[3]
    assert events[3][2] == {
        "model_run_id": 501,
        "manifest_id": "manifest-1",
        "dataset_role": "training",
    }
    assert events[4][2] == {
        "model_run_id": 501,
        "manifest_id": "manifest-1",
        "split_set_id": "manifest-1__kfold_5_seed_42",
        "dataset_role": "training",
        "split_role": "validation",
    }


def test_pipeline_imports_with_split_airflow_package_without_script_import_side_effects(
    tmp_path: Path,
):
    airflow_root = tmp_path / "airflow"
    pricing_root = tmp_path / "pricing"
    airflow_root.mkdir()
    scripts_dir = pricing_root / "scripts"
    scripts_dir.mkdir(parents=True)
    marker_path = tmp_path / "script_imports.txt"
    (scripts_dir / "__init__.py").write_text("", encoding="utf-8")

    shutil.copytree(
        Path("pricing_pipeline"),
        airflow_root / "pricing_pipeline",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (scripts_dir / "load_superglm_excel_to_staging.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "with Path(os.environ['STUB_IMPORT_MARKER']).open('a', encoding='utf-8') as f:\n"
        "    f.write('stage\\n')\n"
        "def stage_rating_export(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (scripts_dir / "load_staging_to_rating_package.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "with Path(os.environ['STUB_IMPORT_MARKER']).open('a', encoding='utf-8') as f:\n"
        "    f.write('publish\\n')\n"
        "def publish_rating_package(*args, **kwargs):\n    return 123\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(airflow_root)
    env["PRICING_PROJECT_ROOT"] = str(pricing_root)
    env["STUB_IMPORT_MARKER"] = str(marker_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pricing_pipeline.orchestration.pipeline; print('pipeline_import=ok')",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "pipeline_import=ok" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
    assert not marker_path.exists()


def test_train_and_export_model_validates_registered_model_without_creating(
    monkeypatch,
    tmp_path: Path,
):
    raw = raw_training_frame()
    model = FakePipelineModel()
    calls = []

    class FakeRun:
        info = SimpleNamespace(run_id="mlflow-run-1")

    class FakeStartRun:
        def __enter__(self):
            return FakeRun()

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda experiment: None,
        start_run=lambda: FakeStartRun(),
        log_param=lambda key, value: calls.append(("log_param", key, value)),
        log_artifact=lambda path, artifact_path=None: None,
        log_metric=lambda key, value, **kwargs: None,
    )

    monkeypatch.setattr(
        pipeline,
        "configure_mlflow",
        lambda uri, **kwargs: fake_mlflow,
    )
    monkeypatch.setattr(pipeline.pd, "read_sql_query", lambda sql, engine: raw)
    monkeypatch.setattr(
        pipeline,
        "ensure_pricing_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("production training must not create model registry rows")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_model_on_engine",
        lambda engine, config: calls.append(("validate_model_on_engine", engine, config)) or 17,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "fit_reml_with_diagnostics",
        lambda model_arg, X, y, *, offset, diagnostics_path, mlflow_client: model_arg,
    )
    monkeypatch.setattr(
        pipeline,
        "export_rating_tables",
        lambda fitted_model, X, y, exposure, *, output_path, mlflow_client: output_path,
    )

    engine = object()
    settings = Settings.from_env(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow.local",
            "RATING_EXPORT_ROOT": str(tmp_path),
        }
    )
    spec = ModelSpec(
        model_name="MTPL_FREQ",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        experiment_name="pricing-mtpl-frequency",
        deployment_slot="MTPL_FREQ_UAT",
        dataset=FREMTPL_DATASET_SPEC,
        training_sql=TRAINING_SQL,
        feature_columns=tuple(FEATURE_COLUMNS),
        build_model=lambda: model,
        build_training_frame=build_training_frame,
    )

    result = pipeline.train_and_export_model(
        engine,
        settings=settings,
        manifest_id="manifest-1",
        split_set_id=None,
        dag_id="pricing_dag",
        airflow_run_id="manual__strict_registry",
        logical_date="2026-05-29",
        spec=spec,
        model_config=MODEL_CONFIG,
        created_by="airflow",
    )

    assert result.model_id == 17
    assert ("validate_model_on_engine", engine, MODEL_CONFIG) in calls
    assert ("log_param", "model_id", 17) in calls


def test_model_export_result_to_dict_omits_unset_publication_receipt_fields(
    tmp_path: Path,
):
    export = ModelExportResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="20260527",
        model_type="superglm_poisson",
        target_name="ClaimNb",
        deployment_slot="MTPL_FREQ_UAT",
        manifest_id="manifest-1",
        dag_id="pricing_dag",
        airflow_run_id="manual__1",
        mlflow_run_id="mlflow-run-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(tmp_path / "rating_tables.xlsx"),
        effective_from="2026-05-27",
        created_by="airflow",
        package_status="PUBLISHED",
    )

    payload = export.to_dict()

    assert "publication_receipt_path" not in payload
    assert "publication_receipt_sha256" not in payload


def test_model_export_result_to_dict_includes_publication_receipt_fields_when_set(
    tmp_path: Path,
):
    export = ModelExportResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="20260527",
        model_type="superglm_poisson",
        target_name="ClaimNb",
        deployment_slot="MTPL_FREQ_UAT",
        manifest_id="manifest-1",
        dag_id="pricing_dag",
        airflow_run_id="manual__1",
        mlflow_run_id="mlflow-run-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(tmp_path / "rating_tables.xlsx"),
        effective_from="2026-05-27",
        created_by="airflow",
        package_status="PUBLISHED",
        publication_receipt_path="/tmp/superglm_publication_receipt.json",
        publication_receipt_sha256="c" * 64,
    )

    payload = export.to_dict()

    assert payload["publication_receipt_path"] == "/tmp/superglm_publication_receipt.json"
    assert payload["publication_receipt_sha256"] == "c" * 64


class FakePipelineModel:
    family = "poisson"

    def __init__(self):
        self.fit_calls = []
        self.result = SimpleNamespace(deviance=7.25)

    def fit_reml(self, X, y, sample_weight=None, offset=None):
        self.fit_calls.append(
            {
                "X": X.copy(),
                "y": y.copy(),
                "sample_weight": None if sample_weight is None else sample_weight.copy(),
                "offset": offset.copy(),
            }
        )
        return self


def test_run_training_export_publish_orchestrates_training_artifacts_and_lineage(
    monkeypatch, tmp_path: Path
):
    raw = raw_training_frame()
    model = FakePipelineModel()
    calls = []

    class FakeRun:
        info = SimpleNamespace(run_id="mlflow-run-1")

    class FakeStartRun:
        def __enter__(self):
            calls.append(("start_run_enter",))
            return FakeRun()

        def __exit__(self, exc_type, exc, tb):
            calls.append(("start_run_exit", exc_type))
            return False

    def fake_log_metric(key, value, **kwargs):
        calls.append(("log_metric", key, value, kwargs))

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda experiment: calls.append(("set_experiment", experiment)),
        start_run=lambda: FakeStartRun(),
        log_param=lambda key, value: calls.append(("log_param", key, value)),
        log_artifact=lambda path, artifact_path=None: calls.append(
            ("log_artifact", path, artifact_path)
        ),
        log_metric=fake_log_metric,
    )

    def fake_read_sql_query(sql, engine):
        calls.append(("read_sql_query", sql, engine))
        return raw

    def fake_export_rating_tables(
        fitted_model,
        X,
        y,
        exposure,
        *,
        output_path,
        mlflow_client,
    ):
        calls.append(
            (
                "export_rating_tables",
                fitted_model,
                X.copy(),
                y.copy(),
                exposure.copy(),
                output_path,
                mlflow_client,
            )
        )
        output_path.write_bytes(b"workbook")
        return output_path

    class FakePublisher:
        def __init__(self, engine_arg, config):
            calls.append(("publisher_init", engine_arg, config))
            self.engine = engine_arg
            self.config = config

        def validate_registered_model(self):
            calls.append(("validate_registered_model",))
            return 17

        def publish_training_export(self, export):
            calls.append(("publish_training_export", export))
            return PublishResult(
                mlflow_run_id=export.mlflow_run_id,
                export_id=export.export_id,
                rate_package_id=123,
                package_version=4,
                rating_workbook_path=export.rating_workbook_path,
            )

    def fake_record_model_run(engine, **kwargs):
        calls.append(("record_model_run", engine, kwargs))

    monkeypatch.setattr(
        pipeline,
        "configure_mlflow",
        lambda uri, **kwargs: calls.append(("configure_mlflow", uri, kwargs)) or fake_mlflow,
    )
    monkeypatch.setattr(pipeline.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(
        pipeline,
        "validate_model_on_engine",
        lambda engine, config: calls.append(("validate_model_on_engine", engine, config)) or 17,
    )
    monkeypatch.setattr(pipeline, "mlflow", fake_mlflow, raising=False)
    monkeypatch.setattr(pipeline, "export_rating_tables", fake_export_rating_tables)
    monkeypatch.setattr(pipeline, "ModelPublisher", FakePublisher, raising=False)
    monkeypatch.setattr(pipeline, "record_model_run", fake_record_model_run)

    dumped = []
    real_dump = pickle.dump

    def fake_dump(obj, handle):
        dumped.append((obj, Path(handle.name)))
        real_dump(obj, handle)

    monkeypatch.setattr(pipeline.pickle, "dump", fake_dump)

    engine = object()
    settings = Settings.from_env(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow.local",
            "RATING_EXPORT_ROOT": str(tmp_path),
        }
    )
    spec = ModelSpec(
        model_name="MTPL_FREQ",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        experiment_name="pricing-mtpl-frequency",
        deployment_slot="MTPL_FREQ_UAT",
        dataset=FREMTPL_DATASET_SPEC,
        training_sql=TRAINING_SQL,
        feature_columns=tuple(FEATURE_COLUMNS),
        build_model=lambda: model,
        build_training_frame=build_training_frame,
    )

    result = pipeline.run_training_export_publish(
        engine,
        settings=settings,
        manifest_id="manifest-1",
        split_set_id="manifest-1__train_test_split_test_0_2_seed_99",
        dag_id="pricing_dag",
        airflow_run_id="scheduled__2026-04-27T10:30:00+00:00",
        logical_date="2026-04-27",
        spec=spec,
        model_config=MODEL_CONFIG,
        created_by="airflow",
    )

    export_id = "mtpl_freq__scheduled__20260427t1030000000"
    workbook_path = tmp_path / "MTPL_FREQ" / "2026-04-27" / export_id / "rating_tables.xlsx"
    model_path = workbook_path.parent / "superglm_model.pkl"

    assert result == {
        "mlflow_run_id": "mlflow-run-1",
        "export_id": export_id,
        "rate_package_id": "123",
        "package_version": "4",
        "package_status": "PUBLISHED",
        "rating_workbook_path": str(workbook_path),
        "was_existing": False,
    }
    assert ("configure_mlflow", "http://mlflow.local", {"enabled": True}) in calls
    assert ("validate_model_on_engine", engine, MODEL_CONFIG) in calls
    assert ("read_sql_query", "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine) in calls
    assert ("set_experiment", "pricing-mtpl-frequency") in calls
    assert ("log_param", "model_name", "MTPL_FREQ") in calls
    assert ("log_param", "model_id", 17) in calls
    assert ("log_param", "model_version", "20260427") in calls
    assert ("log_param", "manifest_id", "manifest-1") in calls
    assert ("log_param", "target", "ClaimNb") in calls
    assert ("log_param", "offset", "log(Exposure)") in calls
    assert ("log_metric", "deviance", 7.25, {}) in calls
    assert ("log_artifact", str(model_path), "model") in calls
    assert (
        "log_artifact",
        str(workbook_path.parent / "superglm_fit.log"),
        "training_diagnostics",
    ) in calls
    assert dumped == [(model, model_path)]

    assert len(model.fit_calls) == 1
    fit_call = model.fit_calls[0]
    assert fit_call["sample_weight"] is None
    np.testing.assert_allclose(fit_call["offset"], np.log(raw["Exposure"]))

    export_call = next(event for event in calls if event[0] == "export_rating_tables")
    assert export_call[1] is model
    np.testing.assert_array_equal(export_call[4], raw["Exposure"].to_numpy(dtype=float))
    assert export_call[5] == workbook_path
    assert export_call[6] is fake_mlflow

    assert ("publisher_init", engine, MODEL_CONFIG) in calls
    assert ("validate_registered_model",) in calls
    publish_call = next(event for event in calls if event[0] == "publish_training_export")
    assert publish_call[1].to_dict() == {
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "20260427",
        "model_type": "superglm_poisson",
        "target_name": "ClaimNb",
        "deployment_slot": "MTPL_FREQ_UAT",
        "manifest_id": "manifest-1",
        "dag_id": "pricing_dag",
        "airflow_run_id": "scheduled__2026-04-27T10:30:00+00:00",
        "mlflow_run_id": "mlflow-run-1",
        "split_set_id": "manifest-1__train_test_split_test_0_2_seed_99",
        "export_id": export_id,
        "rating_workbook_path": str(workbook_path),
        "effective_from": "2026-04-27",
        "created_by": "airflow",
        "package_status": "DRAFT",
    }

    record_call = next(event for event in calls if event[0] == "record_model_run")
    assert record_call == (
        "record_model_run",
        engine,
        {
            "dag_id": "pricing_dag",
            "airflow_run_id": "scheduled__2026-04-27T10:30:00+00:00",
            "mlflow_run_id": "mlflow-run-1",
            "manifest_id": "manifest-1",
            "split_set_id": "manifest-1__train_test_split_test_0_2_seed_99",
            "export_id": export_id,
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_version": "20260427",
            "rate_package_id": 123,
            "rating_workbook_path": str(workbook_path),
            "run_status": "SUCCESS",
            "created_by": "airflow",
        },
    )


def test_run_training_export_publish_continues_when_mlflow_unavailable(
    monkeypatch,
    tmp_path: Path,
):
    raw = raw_training_frame()
    model = FakePipelineModel()
    calls = []

    class NoOpRun:
        info = SimpleNamespace(run_id="")

    class NoOpStartRun:
        def __enter__(self):
            calls.append(("noop_start_run_enter",))
            return NoOpRun()

        def __exit__(self, exc_type, exc, tb):
            calls.append(("noop_start_run_exit", exc_type))
            return False

    class NoOpMlflowClient:
        def set_experiment(self, experiment):
            calls.append(("noop_set_experiment", experiment))

        def start_run(self):
            return NoOpStartRun()

        def log_param(self, key, value):
            calls.append(("noop_log_param", key, value))

        def log_artifact(self, path, artifact_path=None):
            calls.append(("noop_log_artifact", path, artifact_path))

        def log_metric(self, key, value, **kwargs):
            calls.append(("noop_log_metric", key, value, kwargs))

    class RaisingMlflow:
        def __getattr__(self, name):
            raise AssertionError(f"pipeline used global mlflow.{name}")

    def fake_read_sql_query(sql, engine):
        calls.append(("read_sql_query", sql, engine))
        return raw

    def fake_export_rating_tables(
        fitted_model,
        X,
        y,
        exposure,
        *,
        output_path,
        mlflow_client,
    ):
        calls.append(("export_rating_tables", output_path, mlflow_client))
        output_path.write_bytes(b"workbook")
        return output_path

    class FakePublisher:
        def __init__(self, engine_arg, config):
            calls.append(("publisher_init", engine_arg, config))

        def validate_registered_model(self):
            calls.append(("validate_registered_model",))
            return 17

        def publish_training_export(self, export):
            calls.append(("publish_training_export", export))
            return PublishResult(
                mlflow_run_id=export.mlflow_run_id,
                export_id=export.export_id,
                rate_package_id=123,
                package_version=4,
                rating_workbook_path=export.rating_workbook_path,
            )

    monkeypatch.setattr(
        pipeline,
        "configure_mlflow",
        lambda uri, **kwargs: calls.append(("configure_mlflow", uri, kwargs)) or NoOpMlflowClient(),
    )
    monkeypatch.setattr(pipeline, "mlflow", RaisingMlflow(), raising=False)
    monkeypatch.setattr(pipeline.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(
        pipeline,
        "validate_model_on_engine",
        lambda engine, config: calls.append(("validate_model_on_engine", engine, config)) or 17,
    )
    monkeypatch.setattr(pipeline, "export_rating_tables", fake_export_rating_tables)
    monkeypatch.setattr(pipeline, "ModelPublisher", FakePublisher, raising=False)
    monkeypatch.setattr(
        pipeline,
        "record_model_run",
        lambda engine, **kwargs: calls.append(("record_model_run", engine, kwargs)),
    )

    engine = object()
    settings = Settings.from_env(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow.local",
            "RATING_EXPORT_ROOT": str(tmp_path),
        }
    )
    result = pipeline.run_training_export_publish(
        engine,
        settings=settings,
        manifest_id="manifest-1",
        dag_id="pricing_dag",
        airflow_run_id="manual__without_mlflow",
        logical_date="2026-05-28",
        spec=ModelSpec(
            model_name="MTPL_FREQ",
            target_name="ClaimNb",
            model_type="superglm_poisson",
            experiment_name="pricing-mtpl-frequency",
            deployment_slot="MTPL_FREQ_UAT",
            dataset=FREMTPL_DATASET_SPEC,
            training_sql=TRAINING_SQL,
            feature_columns=tuple(FEATURE_COLUMNS),
            build_model=lambda: model,
            build_training_frame=build_training_frame,
        ),
        model_config=MODEL_CONFIG,
        created_by="airflow",
    )

    assert result["mlflow_run_id"] == ""
    assert result["rate_package_id"] == "123"
    assert ("configure_mlflow", "http://mlflow.local", {"enabled": True}) in calls
    assert ("noop_start_run_enter",) in calls
    assert any(call[:2] == ("noop_log_param", "row_count") for call in calls)
    assert any(call[0] == "publish_training_export" for call in calls)
    record_call = next(call for call in calls if call[0] == "record_model_run")
    assert record_call[2]["mlflow_run_id"] == ""


def test_publish_model_export_returns_candidate_without_deploying(
    monkeypatch,
    tmp_path: Path,
):
    calls = []

    class FakePublisher:
        def __init__(self, engine, config):
            calls.append(("publisher_init", engine, config))

        def validate_registered_model(self):
            calls.append(("validate_registered_model",))
            return 17

        def publish_training_export(self, export):
            calls.append(("publish_training_export", export))
            return SimpleNamespace(
                mlflow_run_id="mlflow-run-1",
                export_id=export.export_id if hasattr(export, "export_id") else export["export_id"],
                rate_package_id=123,
                package_version=4,
                package_status="DRAFT",
                was_existing=True,
                rating_workbook_path=(
                    export.rating_workbook_path
                    if hasattr(export, "rating_workbook_path")
                    else export["rating_workbook_path"]
                ),
            )

    monkeypatch.setattr(pipeline, "ModelPublisher", FakePublisher, raising=False)
    monkeypatch.setattr(pipeline, "record_model_run", lambda *args, **kwargs: None)
    engine = object()
    export = {
        "model_id": 17,
        "model_name": "MTPL_FREQ",
        "model_version": "20260527",
        "model_type": "superglm_poisson",
        "target_name": "ClaimNb",
        "deployment_slot": "MTPL_FREQ_UAT",
        "manifest_id": "manifest-1",
        "dag_id": "pricing_dag",
        "airflow_run_id": "manual__1",
        "mlflow_run_id": "mlflow-run-1",
        "split_set_id": None,
        "export_id": "export-1",
        "rating_workbook_path": str(tmp_path / "rating_tables.xlsx"),
        "effective_from": "2026-05-27",
        "created_by": "airflow",
        "package_status": "PUBLISHED",
    }

    result = pipeline.publish_model_export(engine, export, model_config=MODEL_CONFIG)

    assert result["rate_package_id"] == "123"
    assert result["package_version"] == "4"
    assert result["package_status"] == "DRAFT"
    assert result["was_existing"] is True
    assert "publication_receipt_path" not in result
    assert "publication_receipt_sha256" not in result
    assert ("validate_registered_model",) in calls


def test_publish_model_export_includes_publication_receipt_fields_when_set(
    monkeypatch,
    tmp_path: Path,
):
    class FakePublisher:
        def __init__(self, engine, config):
            pass

        def validate_registered_model(self):
            return 17

        def publish_training_export(self, export):
            return SimpleNamespace(
                mlflow_run_id="mlflow-run-1",
                export_id=export.export_id,
                rate_package_id=123,
                package_version=4,
                package_status="DRAFT",
                was_existing=False,
                rating_workbook_path=export.rating_workbook_path,
            )

    monkeypatch.setattr(pipeline, "ModelPublisher", FakePublisher, raising=False)
    monkeypatch.setattr(pipeline, "record_model_run", lambda *args, **kwargs: None)
    engine = object()
    export = ModelExportResult(
        model_id=17,
        model_name="MTPL_FREQ",
        model_version="20260527",
        model_type="superglm_poisson",
        target_name="ClaimNb",
        deployment_slot="MTPL_FREQ_UAT",
        manifest_id="manifest-1",
        dag_id="pricing_dag",
        airflow_run_id="manual__1",
        mlflow_run_id="mlflow-run-1",
        split_set_id=None,
        export_id="export-1",
        rating_workbook_path=str(tmp_path / "rating_tables.xlsx"),
        effective_from="2026-05-27",
        created_by="airflow",
        package_status="PUBLISHED",
        publication_receipt_path="/tmp/superglm_publication_receipt.json",
        publication_receipt_sha256="d" * 64,
    )

    result = pipeline.publish_model_export(engine, export, model_config=MODEL_CONFIG)

    assert result["publication_receipt_path"] == "/tmp/superglm_publication_receipt.json"
    assert result["publication_receipt_sha256"] == "d" * 64
