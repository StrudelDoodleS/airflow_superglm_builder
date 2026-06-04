from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pricing_db import get_runtime, load_env  # noqa: E402
from pricing_pipeline.data.manifest import (  # noqa: E402
    create_dataset_manifest_with_split as create_dataset_manifest,
)
from pricing_pipeline.data.manifest import new_manifest_id  # noqa: E402
from pricing_pipeline.infra.migrations import apply_migrations  # noqa: E402
from pricing_pipeline.orchestration.pipeline import run_training_export_publish  # noqa: E402
from pricing_models.registry import get_model_config, get_model_spec, model_keys  # noqa: E402


def _schema_dir() -> Path:
    path = Path(
        os.environ.get(
            "PRICING_SCHEMA_DIR",
            os.environ.get("PRICING_MIGRATIONS_DIR", "db/migrations"),
        )
    )
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pricing pipeline directly from the host without Airflow or Docker."
    )
    parser.add_argument(
        "--ensure-database",
        action="store_true",
        help="Create the target database if it does not exist.",
    )
    parser.add_argument(
        "--skip-schema-apply",
        action="store_true",
        help="Do not apply schema DDL before the run.",
    )
    parser.add_argument(
        "--skip-raw-load",
        action="store_true",
        help="Do not fetch/load source raw data before creating the manifest.",
    )
    parser.add_argument(
        "--replace-raw",
        action="store_true",
        help="Truncate and reload the model dataset's raw table before training.",
    )
    parser.add_argument(
        "--runtime-module",
        default=None,
        help=(
            "Importable Python module that provides get_engine(database=None), "
            "get_schema_names(), and optional get_runtime_settings()."
        ),
    )
    parser.add_argument("--manifest-id", default=None)
    parser.add_argument(
        "--model-key",
        default="MTPL_FREQ",
        choices=model_keys(),
        help="Registered model spec to train and publish.",
    )
    parser.add_argument("--dag-id", default="no_docker_local")
    parser.add_argument("--airflow-run-id", default=None)
    parser.add_argument("--logical-date", default=None)
    parser.add_argument("--created-by", default="no_docker")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    load_env()
    runtime = get_runtime(args.runtime_module)
    settings = runtime.settings

    if args.ensure_database:
        if settings.skip_database_create:
            print("skip_database_create=true; not creating database")
        else:
            runtime.ensure_database(settings.pricing_database)

    engine = runtime.get_engine()
    model_spec = get_model_spec(args.model_key)
    model_config = get_model_config(args.model_key)

    if not args.skip_schema_apply:
        applied = apply_migrations(engine, _schema_dir())
        print(f"schema_files_applied={len(applied)}")

    if not args.skip_raw_load:
        if model_spec.dataset.raw_loader is None:
            print(f"{model_spec.dataset.dataset_name}_raw_loader=none")
        else:
            raw_rows = model_spec.dataset.raw_loader(engine, replace=args.replace_raw)
            print(f"{model_spec.dataset.dataset_name}_raw_rows={raw_rows}")

    manifest_id = args.manifest_id or new_manifest_id(model_spec.dataset.dataset_name)
    manifest_result = create_dataset_manifest(
        engine,
        dataset=model_spec.dataset,
        manifest_id=manifest_id,
        validation_split=model_config.validation_split,
        validation_split_artifact_root=settings.validation_split_artifact_root,
        created_by=args.created_by,
    )
    logical_date = args.logical_date or datetime.now(UTC).date().isoformat()
    airflow_run_id = args.airflow_run_id or f"manual__{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    result = run_training_export_publish(
        engine,
        settings=settings,
        manifest_id=manifest_result.manifest_id,
        split_set_id=manifest_result.split_set_id,
        dag_id=args.dag_id,
        airflow_run_id=airflow_run_id,
        logical_date=logical_date,
        spec=model_spec,
        model_config=model_config,
        created_by=args.created_by,
    )
    print(json.dumps({"manifest_id": manifest_result.manifest_id, **result}, indent=2))


if __name__ == "__main__":
    main()
