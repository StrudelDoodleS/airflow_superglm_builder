from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pricing_pipeline.infra.config import Settings
from pricing_pipeline.models.config import ModelBuildConfig
from pricing_pipeline.models.spec import DatasetSpec, ModelSpec, TrainingFrame
from pricing_pipeline.orchestration import pipeline
from pricing_pipeline.publishing.lifecycle import (
    DeploymentResult,
    PublishResult,
    RatePackageSelector,
    RatePackageSnapshot,
)
from pricing_pipeline.publishing.publisher import ModelPublisher


class FakeWorkflowModel:
    result = SimpleNamespace(deviance=1.25)


class FakeRunContext:
    def __init__(self, run_id: str):
        self.run = SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def __enter__(self):
        return self.run

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMlflow:
    def __init__(self):
        self.params = []
        self.artifacts = []
        self.metrics = []
        self.run_count = 0

    def set_experiment(self, experiment_name: str) -> None:
        self.params.append(("experiment", experiment_name))

    def start_run(self):
        self.run_count += 1
        return FakeRunContext(f"mlflow-run-{self.run_count}")

    def log_param(self, key, value) -> None:
        self.params.append((key, value))

    def log_metric(self, key, value, **kwargs) -> None:
        self.metrics.append((key, value, kwargs))

    def log_artifact(self, path, artifact_path=None) -> None:
        self.artifacts.append((path, artifact_path))


class LifecycleState:
    def __init__(self):
        self.stage_calls = []
        self.packages = {}
        self.current_by_slot = {}
        self.deployments = []
        self.recorded_runs = []
        self.manual_diffs = []

    def package_by_selector(
        self,
        *,
        rate_package_id: int | None = None,
        package_version: int | None = None,
    ) -> dict:
        if rate_package_id is not None:
            return self.packages[int(rate_package_id)]
        for package in self.packages.values():
            if package["package_version"] == package_version:
                return package
        raise AssertionError(f"missing package_version={package_version}")


def config() -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
    )


def spec() -> ModelSpec:
    dataset = DatasetSpec(
        dataset_name="freMTPL",
        source_system="unit-test",
        manifest_sql="SELECT * FROM raw_policy",
        pk_columns=("policy_id",),
        target_column="ClaimNb",
    )

    def build_training_frame(raw: pd.DataFrame) -> TrainingFrame:
        exposure = raw["Exposure"].to_numpy(dtype=float)
        return TrainingFrame(
            X=raw[["Area"]],
            y=raw["ClaimNb"].to_numpy(dtype=float),
            exposure=exposure,
            offset=np.log(exposure),
        )

    return ModelSpec(
        model_name="MTPL_FREQ",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        experiment_name="pricing-mtpl-frequency",
        deployment_slot="MTPL_FREQ_UAT",
        dataset=dataset,
        training_sql="SELECT * FROM raw_policy",
        feature_columns=("Area",),
        build_model=FakeWorkflowModel,
        build_training_frame=build_training_frame,
        model_label="Motor frequency",
    )


def rate_cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [1, 2],
            "term_id": [10, 10],
            "cell_key_text": ["Area=A", "Area=B"],
            "cell_key_digest": ["digest-a", "digest-b"],
            "is_reference": [True, False],
            "is_default": [False, False],
            "multiplier": [1.0, 1.20],
            "log_coefficient": [0.0, float(np.log(1.20))],
        }
    )


def snapshot_for(package: dict) -> RatePackageSnapshot:
    return RatePackageSnapshot(
        metadata={
            "rate_package_id": package["rate_package_id"],
            "parent_rate_package_id": package.get("parent_rate_package_id"),
            "model_id": 17,
            "model_name": "MTPL_FREQ",
            "model_version": package["model_version"],
            "package_version": package["package_version"],
            "base_rate": 1.0,
            "effective_from_date": "2026-05-01",
            "effective_to_date": None,
            "package_status": "PUBLISHED",
            "created_by": "airflow",
        },
        terms=pd.DataFrame(),
        rate_cells=rate_cells(),
        cell_levels=pd.DataFrame(),
        compiled_rate_cells=pd.DataFrame(),
        compiled_1d_bands=pd.DataFrame(),
    )


def test_build_deploy_retrain_deploy_manual_uplift_deploy_workflow(
    monkeypatch,
    tmp_path,
):
    engine = object()
    state = LifecycleState()
    fake_mlflow = FakeMlflow()
    training_frames = [
        pd.DataFrame(
            {
                "policy_id": [1, 2],
                "Area": ["A", "B"],
                "ClaimNb": [0, 1],
                "Exposure": [0.5, 1.0],
            }
        ),
        pd.DataFrame(
            {
                "policy_id": [1, 2, 3, 4],
                "Area": ["A", "B", "B", "A"],
                "ClaimNb": [0, 1, 0, 1],
                "Exposure": [0.5, 1.0, 0.75, 1.25],
            }
        ),
    ]

    def fake_stage_rating_export(engine_arg, **kwargs):
        assert engine_arg is engine
        state.stage_calls.append(kwargs)

    def fake_publish_rating_package(engine_arg, **kwargs):
        assert engine_arg is engine
        package_version = len(state.packages) + 1
        rate_package_id = 100 + package_version
        stage_call = state.stage_calls[-1]
        package = {
            "rate_package_id": rate_package_id,
            "package_version": package_version,
            "model_version": stage_call["model_version"],
            "package_status": kwargs["package_status"],
            "export_id": kwargs["export_id"],
            "parent_rate_package_id": None,
        }
        state.packages[rate_package_id] = package
        return PublishResult(
            mlflow_run_id="",
            export_id=kwargs["export_id"],
            rate_package_id=rate_package_id,
            package_version=package_version,
            rating_workbook_path=str(stage_call["workbook_path"]),
        )

    def fake_deploy_rate_package(engine_arg, config_arg, **kwargs):
        assert engine_arg is engine
        assert config_arg == config()
        package = state.package_by_selector(
            rate_package_id=kwargs["rate_package_id"],
            package_version=kwargs["package_version"],
        )
        assert package["package_status"] == "PUBLISHED"
        slot = (kwargs["deployment_slot"] or config_arg.deployment_slot).strip().upper()
        previous = state.current_by_slot.get(slot)
        assert kwargs["expected_current_rate_package_id"] == previous
        state.current_by_slot[slot] = package["rate_package_id"]
        result = DeploymentResult(
            model_id=kwargs["model_id"],
            deployment_slot=slot,
            previous_rate_package_id=previous,
            rate_package_id=package["rate_package_id"],
            package_version=package["package_version"],
            deployed_by=kwargs["deployed_by"].strip(),
            deployment_reason=kwargs["deployment_reason"].strip(),
        )
        state.deployments.append(result)
        return result

    def fake_load_rate_package_snapshot(engine_arg, config_arg, selector):
        assert engine_arg is engine
        assert config_arg == config()
        package = state.package_by_selector(
            rate_package_id=selector.rate_package_id,
            package_version=selector.package_version,
        )
        return snapshot_for(package)

    def fake_write_manual_revision(
        engine_arg,
        config_arg,
        *,
        parent,
        edited_rate_cells,
        diff,
        reason,
        created_by,
    ):
        assert engine_arg is engine
        assert config_arg == config()
        assert parent.metadata["rate_package_id"] == 102
        assert reason == "manual uplift Area=B"
        assert created_by == "pricing-user"
        state.manual_diffs.append(diff)
        rate_package_id = 103
        state.packages[rate_package_id] = {
            "rate_package_id": rate_package_id,
            "package_version": 3,
            "model_version": parent.metadata["model_version"],
            "package_status": "PUBLISHED",
            "export_id": "manual-uplift",
            "parent_rate_package_id": parent.metadata["rate_package_id"],
        }
        return rate_package_id, 3

    def fake_read_sql_query(sql, engine_arg):
        assert engine_arg is engine
        assert "raw_policy" in str(sql)
        return training_frames.pop(0)

    def fake_fit_reml_with_diagnostics(model, X, y, *, offset, diagnostics_path, mlflow_client):
        diagnostics_path.write_text("diagnostics", encoding="utf-8")
        return model

    def fake_export_rating_tables(
        fitted_model,
        X,
        y,
        exposure,
        *,
        output_path,
        mlflow_client,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rating workbook")
        return output_path

    monkeypatch.setattr(pipeline, "configure_mlflow", lambda uri, **kwargs: fake_mlflow)
    monkeypatch.setattr(pipeline, "validate_model_on_engine", lambda *args, **kwargs: 17)
    monkeypatch.setattr(pipeline.pd, "read_sql_query", fake_read_sql_query)
    monkeypatch.setattr(pipeline, "fit_reml_with_diagnostics", fake_fit_reml_with_diagnostics)
    monkeypatch.setattr(pipeline, "export_rating_tables", fake_export_rating_tables)
    monkeypatch.setattr(
        pipeline,
        "record_model_run",
        lambda engine_arg, **kwargs: state.recorded_runs.append(kwargs),
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
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.deploy_rate_package",
        fake_deploy_rate_package,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.publisher.load_rate_package_snapshot",
        fake_load_rate_package_snapshot,
    )
    monkeypatch.setattr(
        "pricing_pipeline.publishing.manual_revision._write_manual_revision",
        fake_write_manual_revision,
    )

    settings = Settings.from_env(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow.local",
            "RATING_EXPORT_ROOT": str(tmp_path),
        }
    )
    publisher = ModelPublisher(engine, config())

    first_build = pipeline.run_training_export_publish(
        engine,
        settings=settings,
        manifest_id="manifest-v1",
        dag_id="pricing_model_build",
        airflow_run_id="manual__v1",
        logical_date="2026-05-01",
        spec=spec(),
        model_config=config(),
        created_by="airflow",
    )
    first_deploy = publisher.deploy(
        rate_package_id=int(first_build["rate_package_id"]),
        expected_current_rate_package_id=None,
        deployment_reason="initial approval",
        deployed_by="airflow",
    )

    second_build = pipeline.run_training_export_publish(
        engine,
        settings=settings,
        manifest_id="manifest-v2",
        dag_id="pricing_model_build",
        airflow_run_id="manual__v2",
        logical_date="2026-05-08",
        spec=spec(),
        model_config=config(),
        created_by="airflow",
    )
    second_deploy = publisher.deploy(
        rate_package_id=int(second_build["rate_package_id"]),
        expected_current_rate_package_id=101,
        deployment_reason="more data approval",
        deployed_by="airflow",
    )

    parent = publisher.load_rate_package(
        RatePackageSelector(rate_package_id=int(second_build["rate_package_id"]))
    )
    edited = parent.rate_cells.copy()
    edited.loc[edited["cell_id"] == 2, "multiplier"] = 1.32
    manual_revision = publisher.create_manual_revision(
        parent=parent,
        edited_rate_cells=edited,
        reason="manual uplift Area=B",
        created_by="pricing-user",
    )
    manual_deploy = publisher.deploy(
        rate_package_id=manual_revision.rate_package_id,
        expected_current_rate_package_id=102,
        deployment_reason="manual uplift approval",
        deployed_by="pricing-user",
    )

    assert first_build["rate_package_id"] == "101"
    assert first_build["package_version"] == "1"
    assert first_deploy.previous_rate_package_id is None
    assert second_build["rate_package_id"] == "102"
    assert second_build["package_version"] == "2"
    assert second_deploy.previous_rate_package_id == 101
    assert manual_revision.rate_package_id == 103
    assert manual_revision.package_version == 3
    assert manual_revision.parent_rate_package_id == 102
    assert manual_deploy.previous_rate_package_id == 102
    assert state.current_by_slot["MTPL_FREQ_UAT"] == 103

    assert [call["model_version"] for call in state.stage_calls] == [
        "20260501",
        "20260508",
    ]
    assert [value for key, value in fake_mlflow.params if key == "row_count"] == [2, 4]
    assert [run["rate_package_id"] for run in state.recorded_runs] == [101, 102]
    assert [
        (deployment.rate_package_id, deployment.previous_rate_package_id)
        for deployment in state.deployments
    ] == [(101, None), (102, 101), (103, 102)]

    diff = state.manual_diffs[0]
    assert diff["cell_id"].tolist() == [2]
    np.testing.assert_allclose(diff["old_multiplier"], [1.20])
    np.testing.assert_allclose(diff["new_multiplier"], [1.32])
