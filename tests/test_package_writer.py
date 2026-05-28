from pathlib import Path

import pytest

from pricing_pipeline.publishing.lifecycle import PublishResult
from pricing_pipeline.publishing.package_writer import load_staging_to_rating_package
from pricing_pipeline.publishing.package_writer import publish_rating_package


def test_package_writer_rejects_legacy_pointer_deployment():
    args = type(
        "Args",
        (),
        {
            "export_id": "export-1",
            "created_by": "airflow",
            "package_status": "PUBLISHED",
            "set_pointer": "MTPL_FREQ_UAT",
        },
    )()

    with pytest.raises(ValueError, match="deploy"):
        load_staging_to_rating_package(object(), args)


def test_package_writer_does_not_write_deployment_tables_during_publish():
    writer = Path("pricing_pipeline/publishing/package_writer.py").read_text(
        encoding="utf-8"
    )

    assert "PRICING_MODEL_DEPLOYMENT" not in writer
    assert "PRICING_PACKAGE_POINTER" not in writer


def test_publish_rating_package_builds_args_without_deployment_pointer(monkeypatch):
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

    result = publish_rating_package(
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
    args = captured[0][1]
    assert args.export_id == "export-1"
    assert args.created_by == "airflow"
    assert args.package_status == "PUBLISHED"
    assert args.set_pointer is None
