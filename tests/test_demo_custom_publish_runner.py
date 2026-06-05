from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


def test_run_demo_custom_publish_registers_model_and_publishes(monkeypatch, tmp_path):
    from scripts import run_demo_custom_publish as runner

    engine = object()
    calls = []
    completed_build = {
        "rating_workbook_path": str(tmp_path / "rating_tables.xlsx"),
        "model_version": "v7",
        "effective_from": "2026-06-04",
        "created_by": "pytest",
        "export_id": "demo_custom_freq__python__20260604",
    }
    publish_result = SimpleNamespace(
        to_dict=lambda: {
            "rate_package_id": 101,
            "package_version": 7,
            "package_status": "PUBLISHED",
        }
    )

    monkeypatch.setattr(
        runner,
        "get_runtime",
        lambda runtime_module=None: SimpleNamespace(
            get_engine=lambda: engine,
            settings=SimpleNamespace(validation_split_artifact_root=tmp_path / "splits"),
        ),
    )
    monkeypatch.setattr(
        runner,
        "ensure_pricing_model",
        lambda engine_arg, **kwargs: (
            calls.append(("ensure_pricing_model", engine_arg, kwargs)) or 17
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_demo_training_frame",
        lambda: pd.DataFrame({"policy_id": [1], "claim_count": [0], "exposure": [1.0]}),
    )
    monkeypatch.setattr(runner, "build_final_model_frame", lambda frame: frame)
    monkeypatch.setattr(
        runner,
        "materialize_training_source",
        lambda engine_arg, frame, *, table_name=None: (
            calls.append(("materialize_training_source", engine_arg, len(frame), table_name))
            or len(frame)
        ),
    )
    monkeypatch.setattr(
        runner,
        "write_training_frame",
        lambda frame, output_dir: (
            calls.append(("write_training_frame", len(frame), output_dir))
            or str(tmp_path / "training_frame.csv")
        ),
    )
    monkeypatch.setattr(
        runner,
        "trained_model_version_for_export",
        lambda engine_arg, *, model_key, export_id: (
            calls.append(
                (
                    "trained_model_version_for_export",
                    engine_arg,
                    model_key,
                    export_id,
                )
            )
            or "v7"
        ),
    )
    monkeypatch.setattr(
        runner,
        "effective_from_for_run",
        lambda: calls.append(("effective_from_for_run",)) or "2026-06-04",
    )
    monkeypatch.setattr(
        runner,
        "export_superglm_completed_build",
        lambda frame, **kwargs: (
            calls.append(("export_superglm_completed_build", len(frame), kwargs)) or completed_build
        ),
    )
    monkeypatch.setattr(
        runner,
        "create_model_frame_manifest_with_split",
        lambda engine_arg, **kwargs: (
            calls.append(("create_model_frame_manifest_with_split", engine_arg, kwargs))
            or SimpleNamespace(manifest_id="manifest-1", split_set_id="split-1")
        ),
    )
    monkeypatch.setattr(
        runner,
        "publish_completed_model_build",
        lambda engine_arg, **kwargs: (
            calls.append(("publish_completed_model_build", engine_arg, kwargs)) or publish_result
        ),
    )

    result = runner.run_demo_custom_publish(output_dir=tmp_path, created_by="pytest")

    assert result["rate_package_id"] == 101
    assert calls[0][0] == "ensure_pricing_model"
    assert calls[0][2]["model_key"] == "DEMO_CUSTOM_FREQ"
    assert calls[0][2]["created_by"] == "pytest"
    assert [call[0] for call in calls] == [
        "ensure_pricing_model",
        "effective_from_for_run",
        "trained_model_version_for_export",
        "materialize_training_source",
        "write_training_frame",
        "export_superglm_completed_build",
        "create_model_frame_manifest_with_split",
        "publish_completed_model_build",
    ]
    materialize_call = calls[3]
    assert materialize_call[3].startswith("DEMO_CUSTOM_PUBLISH_TRAINING_")
    manifest_call = calls[-2]
    assert manifest_call[2]["frame"].equals(
        pd.DataFrame({"policy_id": [1], "claim_count": [0], "exposure": [1.0]})
    )
    assert manifest_call[2]["spec"].dataset_name == "demo_custom_frequency_model_frame"
    assert manifest_call[2]["spec"].data_as_of_date.isoformat() == "2026-06-04"
    version_call = calls[2]
    assert version_call[2] == "DEMO_CUSTOM_FREQ"
    assert version_call[3] == "demo_custom_freq__python__20260604"
    export_call = calls[-3]
    assert export_call[2]["model_version"] == "v7"
    assert export_call[2]["effective_from"] == "2026-06-04"
    assert export_call[2]["export_id"] == "demo_custom_freq__python__20260604"
    publish_call = calls[-1]
    assert publish_call[2]["settings"].validation_split_artifact_root == tmp_path / "splits"
    assert publish_call[2]["completed_build"]["manifest_id"] == "manifest-1"
    assert publish_call[2]["completed_build"]["split_set_id"] == "split-1"
    assert publish_call[2]["created_by"] == "pytest"


def test_run_demo_custom_publish_reads_default_output_dir_from_env(monkeypatch, tmp_path):
    from scripts import run_demo_custom_publish as runner

    monkeypatch.setenv("PRICING_DEMO_CUSTOM_OUTPUT_DIR", str(tmp_path / "demo-output"))

    assert runner.default_output_dir() == tmp_path / "demo-output"
