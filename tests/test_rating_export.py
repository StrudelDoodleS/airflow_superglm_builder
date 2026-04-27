from __future__ import annotations

import pickle
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pricing_pipeline import lineage, pipeline, rating_export, rating_package
from pricing_pipeline.config import Settings
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
    export_id = rating_export.build_export_id(
        "MTPL_FREQ", "scheduled__2026-04-27T10:30:00+00:00"
    )

    assert export_id == "mtpl_freq__scheduled__20260427t1030000000"


def test_build_rating_export_path_uses_model_and_date(tmp_path: Path):
    path = rating_export.build_rating_export_path(
        tmp_path,
        model_name="MTPL_FREQ",
        logical_date="2026-04-27",
        export_id="mtpl_freq__run1",
    )

    assert (
        path
        == tmp_path / "MTPL_FREQ" / "2026-04-27" / "mtpl_freq__run1" / "rating_tables.xlsx"
    )


def test_build_rating_export_path_accepts_positional_call_shape(tmp_path: Path):
    path = rating_export.build_rating_export_path(
        tmp_path, "MTPL_FREQ", "2026-04-27", "mtpl_freq__run1"
    )

    assert (
        path
        == tmp_path / "MTPL_FREQ" / "2026-04-27" / "mtpl_freq__run1" / "rating_tables.xlsx"
    )


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
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append(
            (path, artifact_path)
        )
    )
    monkeypatch.setattr(rating_export, "mlflow", fake_mlflow)

    output_path = tmp_path / "nested" / "rating_tables.xlsx"
    X = pd.DataFrame({"x": [1, 2]})
    y = np.array([0.0, 1.0])
    exposure = np.array([0.5, 2.0])

    result = rating_export.export_rating_tables(
        model, X, y, exposure, output_path=output_path
    )

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"
    assert model.calls == [
        ((output_path, X, y), {"sample_weight": exposure, "n_bins": 150})
    ]
    assert mlflow_calls == [(str(output_path), "rating_tables")]


def test_export_rating_tables_accepts_positional_output_path(monkeypatch, tmp_path: Path):
    model = FakeExportModel()
    mlflow_calls = []
    fake_mlflow = SimpleNamespace(
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append(
            (path, artifact_path)
        )
    )
    monkeypatch.setattr(rating_export, "mlflow", fake_mlflow)

    output_path = tmp_path / "nested" / "rating_tables.xlsx"
    X = pd.DataFrame({"x": [1, 2]})
    y = np.array([0.0, 1.0])
    exposure = np.array([0.5, 2.0])

    result = rating_export.export_rating_tables(model, X, y, exposure, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"workbook"
    assert model.calls == [
        ((output_path, X, y), {"sample_weight": exposure, "n_bins": 150})
    ]
    assert mlflow_calls == [(str(output_path), "rating_tables")]


def test_export_rating_tables_requires_superglm_rating_export_support(
    monkeypatch, tmp_path: Path
):
    mlflow_calls = []
    fake_mlflow = SimpleNamespace(
        log_artifact=lambda path, artifact_path=None: mlflow_calls.append(
            (path, artifact_path)
        )
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
    def __init__(self, label: str, events: list[tuple]):
        self.label = label
        self.events = events

    def to_sql(self, name, con, schema=None, if_exists=None, index=None, chunksize=None):
        self.events.append(
            ("to_sql", self.label, name, con, schema, if_exists, index, chunksize)
        )


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
    def __init__(self, events: list[tuple]):
        self.events = events
        self.connection = FakeBeginConnection(events)

    def begin(self):
        self.events.append(("begin",))
        return FakeBegin(self.connection)


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

    monkeypatch.setattr(
        load_superglm_excel_to_staging,
        "build_staging_frames",
        fake_build_staging_frames,
    )
    engine = FakeEngine(events)
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
    )

    args = captured_args[0]
    assert args.xlsx == workbook_path
    assert args.sheet == "Rating Tables"
    assert args.export_id == "export-1"
    assert args.model_name == "MTPL_FREQ"
    assert args.model_version == "20260427"
    assert args.effective_from == "2026-04-27"
    assert args.effective_to is None
    assert args.created_by == "airflow"
    assert args.replace is True

    delete_sql = [event[1] for event in events if event[0] == "execute"]
    assert delete_sql == [
        "DELETE FROM pricing.STG_CELL_LEVEL WHERE export_id = :export_id",
        "DELETE FROM pricing.STG_RATE_CELL WHERE export_id = :export_id",
        "DELETE FROM pricing.STG_RATING_EXPORT WHERE export_id = :export_id",
    ]
    assert [event[2] for event in events if event[0] == "execute"] == [
        {"export_id": "export-1"},
        {"export_id": "export-1"},
        {"export_id": "export-1"},
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


def test_publish_rating_package_wrapper_passes_args_and_returns_package_id(monkeypatch):
    calls = []

    def fake_publish(engine, **kwargs):
        calls.append((engine, kwargs))
        return 42

    monkeypatch.setattr(rating_package, "_publish", fake_publish)
    engine = object()

    package_id = rating_package.publish_rating_package(
        engine,
        export_id="export-1",
        pointer_name="MTPL_FREQ_UAT",
        created_by="airflow",
        package_status="PUBLISHED",
    )

    assert package_id == 42
    assert calls == [
        (
            engine,
            {
                "export_id": "export-1",
                "pointer_name": "MTPL_FREQ_UAT",
                "created_by": "airflow",
                "package_status": "PUBLISHED",
            },
        )
    ]


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
        pointer_name=None,
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


def test_record_model_run_uses_parameterized_sql_with_expected_params():
    events = []
    engine = FakeEngine(events)

    lineage.record_model_run(
        engine,
        dag_id="dag",
        airflow_run_id="scheduled__2026-04-27",
        mlflow_run_id="mlflow-1",
        manifest_id="manifest-1",
        export_id="export-1",
        model_name="MTPL_FREQ",
        model_version="20260427",
        rate_package_id=42,
        rating_workbook_path="/tmp/rating_tables.xlsx",
        run_status="SUCCESS",
        created_by="airflow",
    )

    assert len(events) == 2
    sql = events[1][1]
    params = events[1][2]
    assert "MERGE pricing.MODEL_RUN" in sql
    assert "WHEN MATCHED THEN" in sql
    assert "WHEN NOT MATCHED THEN" in sql
    assert "tgt.dag_id = src.dag_id" in sql
    assert "tgt.airflow_run_id = src.airflow_run_id" in sql
    assert "tgt.model_name = src.model_name" in sql
    assert "SYSUTCDATETIME()" in sql
    assert ":dag_id" in sql
    assert params == {
        "dag_id": "dag",
        "airflow_run_id": "scheduled__2026-04-27",
        "mlflow_run_id": "mlflow-1",
        "manifest_id": "manifest-1",
        "export_id": "export-1",
        "model_name": "MTPL_FREQ",
        "model_version": "20260427",
        "rate_package_id": 42,
        "rating_workbook_path": "/tmp/rating_tables.xlsx",
        "run_status": "SUCCESS",
        "created_by": "airflow",
    }


def test_pipeline_imports_with_split_airflow_package_and_pricing_scripts_paths(
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
            "import pricing_pipeline.pipeline; print('pipeline_import=ok')",
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
    assert marker_path.read_text(encoding="utf-8").splitlines() == ["publish", "stage"]


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
                "sample_weight": sample_weight.copy(),
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

    fake_mlflow = SimpleNamespace(
        set_experiment=lambda experiment: calls.append(("set_experiment", experiment)),
        start_run=lambda: FakeStartRun(),
        log_param=lambda key, value: calls.append(("log_param", key, value)),
        log_artifact=lambda path, artifact_path=None: calls.append(
            ("log_artifact", path, artifact_path)
        ),
        log_metric=lambda key, value: calls.append(("log_metric", key, value)),
    )

    def fake_read_sql_query(sql, engine):
        calls.append(("read_sql_query", sql, engine))
        return raw

    def fake_export_rating_tables(fitted_model, X, y, exposure, *, output_path):
        calls.append(
            ("export_rating_tables", fitted_model, X.copy(), y.copy(), exposure.copy(), output_path)
        )
        output_path.write_bytes(b"workbook")
        return output_path

    def fake_stage_rating_export(engine, **kwargs):
        calls.append(("stage_rating_export", engine, kwargs))

    def fake_publish_rating_package(engine, **kwargs):
        calls.append(("publish_rating_package", engine, kwargs))
        return 123

    def fake_record_model_run(engine, **kwargs):
        calls.append(("record_model_run", engine, kwargs))

    monkeypatch.setattr(pipeline, "configure_mlflow", lambda uri: calls.append(("configure_mlflow", uri)))
    monkeypatch.setattr(pipeline.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(pipeline, "build_model", lambda: model)
    monkeypatch.setattr(pipeline, "mlflow", fake_mlflow)
    monkeypatch.setattr(pipeline, "export_rating_tables", fake_export_rating_tables)
    monkeypatch.setattr(pipeline, "stage_rating_export", fake_stage_rating_export)
    monkeypatch.setattr(pipeline, "publish_rating_package", fake_publish_rating_package)
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

    result = pipeline.run_training_export_publish(
        engine,
        settings=settings,
        manifest_id="manifest-1",
        dag_id="pricing_dag",
        airflow_run_id="scheduled__2026-04-27T10:30:00+00:00",
        logical_date="2026-04-27",
        created_by="airflow",
    )

    export_id = "mtpl_freq__scheduled__20260427t1030000000"
    workbook_path = (
        tmp_path / "MTPL_FREQ" / "2026-04-27" / export_id / "rating_tables.xlsx"
    )
    model_path = workbook_path.parent / "superglm_model.pkl"

    assert result == {
        "mlflow_run_id": "mlflow-run-1",
        "export_id": export_id,
        "rate_package_id": "123",
        "rating_workbook_path": str(workbook_path),
    }
    assert ("configure_mlflow", "http://mlflow.local") in calls
    assert ("read_sql_query", "SELECT * FROM pricing.FREMTPL_RAW ORDER BY IDpol", engine) in calls
    assert ("set_experiment", "pricing-mtpl-frequency") in calls
    assert ("log_param", "model_name", "MTPL_FREQ") in calls
    assert ("log_param", "model_version", "20260427") in calls
    assert ("log_param", "manifest_id", "manifest-1") in calls
    assert ("log_param", "target", "ClaimNb") in calls
    assert ("log_param", "offset", "log(Exposure)") in calls
    assert ("log_metric", "deviance", 7.25) in calls
    assert ("log_artifact", str(model_path), "model") in calls
    assert dumped == [(model, model_path)]

    assert len(model.fit_calls) == 1
    fit_call = model.fit_calls[0]
    np.testing.assert_array_equal(
        fit_call["sample_weight"], raw["Exposure"].to_numpy(dtype=float)
    )
    np.testing.assert_allclose(fit_call["offset"], np.log(raw["Exposure"]))

    export_call = next(event for event in calls if event[0] == "export_rating_tables")
    assert export_call[1] is model
    np.testing.assert_array_equal(export_call[4], raw["Exposure"].to_numpy(dtype=float))
    assert export_call[5] == workbook_path

    stage_call = next(event for event in calls if event[0] == "stage_rating_export")
    assert stage_call == (
        "stage_rating_export",
        engine,
        {
            "workbook_path": workbook_path,
            "export_id": export_id,
            "model_name": "MTPL_FREQ",
            "model_version": "20260427",
            "effective_from": "2026-04-27",
            "created_by": "airflow",
            "replace": True,
        },
    )

    publish_call = next(event for event in calls if event[0] == "publish_rating_package")
    assert publish_call == (
        "publish_rating_package",
        engine,
        {
            "export_id": export_id,
            "pointer_name": "MTPL_FREQ_UAT",
            "created_by": "airflow",
            "package_status": "DRAFT",
        },
    )

    record_call = next(event for event in calls if event[0] == "record_model_run")
    assert record_call == (
        "record_model_run",
        engine,
        {
            "dag_id": "pricing_dag",
            "airflow_run_id": "scheduled__2026-04-27T10:30:00+00:00",
            "mlflow_run_id": "mlflow-run-1",
            "manifest_id": "manifest-1",
            "export_id": export_id,
            "model_name": "MTPL_FREQ",
            "model_version": "20260427",
            "rate_package_id": 123,
            "rating_workbook_path": str(workbook_path),
            "run_status": "SUCCESS",
            "created_by": "airflow",
        },
    )
