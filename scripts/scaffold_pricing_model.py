from __future__ import annotations

import argparse
import keyword
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ScaffoldOptions:
    model_key: str
    target_name: str
    model_label: str | None = None
    model_type: str = "superglm_poisson"
    deployment_slot: str | None = None
    package_name: str | None = None
    dag_id: str | None = None
    experiment_name: str | None = None
    template: str = "custom"
    root: Path = Path(".")
    force: bool = False


@dataclass(frozen=True)
class ScaffoldResult:
    package_name: str
    dag_id: str
    created_files: tuple[Path, ...]


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _required_text(value: str | None, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _validate_model_key(model_key: str) -> str:
    cleaned = _required_text(model_key, "model_key")
    if not _MODEL_KEY.match(cleaned):
        raise ValueError(
            "model_key must start with a letter and contain only letters, numbers, and underscores"
        )
    return cleaned


def _module_name_from_model_key(model_key: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", model_key).strip("_").lower()
    value = re.sub(r"_+", "_", value)
    if not value:
        raise ValueError("model_key must produce a non-empty package name")
    return value


def _validate_python_identifier(value: str, field_name: str) -> str:
    cleaned = _required_text(value, field_name)
    if not _PYTHON_IDENTIFIER.match(cleaned) or keyword.iskeyword(cleaned):
        raise ValueError(f"{field_name} must be a valid Python identifier")
    return cleaned


def _default_label(model_key: str) -> str:
    return model_key.replace("_", " ").title()


def _model_toml_template(
    *,
    model_key: str,
    model_label: str,
    target_name: str,
    model_type: str,
    deployment_slot: str,
) -> str:
    return dedent(
        f"""\
        model_key = {_toml_string(model_key)}
        model_label = {_toml_string(model_label)}
        target_name = {_toml_string(target_name)}
        model_type = {_toml_string(model_type)}
        deployment_slot = {_toml_string(deployment_slot)}
        default_package_status = "PUBLISHED"

        [validation_split]
        method = "train_test_split"
        test_size = 0.20
        random_state = 42
        shuffle = true
        materialize = true
        """
    )


def _training_template(*, target_name: str) -> str:
    return dedent(
        f"""\
        from __future__ import annotations

        import numpy as np
        import pandas as pd

        from pricing_pipeline.models.spec import TrainingFrame


        TRAINING_SQL = \"\"\"
        SELECT *
        FROM your_schema.your_training_view
        \"\"\"

        FEATURE_COLUMNS: list[str] = [
            # "rating_factor",
        ]


        def build_training_frame(raw: pd.DataFrame) -> TrainingFrame:
            df = raw.copy()

            # Create derived targets here when the source SQL/view is read-only.
            # df[{_toml_string(target_name)}] = df["numerator"] / df["denominator"]

            missing = [
                column
                for column in [*FEATURE_COLUMNS, {_toml_string(target_name)}]
                if column not in df.columns
            ]
            if missing:
                raise ValueError(f"missing columns: {{', '.join(missing)}}")

            X = df.loc[:, FEATURE_COLUMNS].copy()
            y = df[{_toml_string(target_name)}].to_numpy(dtype=float)
            exposure = np.ones(len(df), dtype=float)
            offset = np.zeros(len(df), dtype=float)
            return TrainingFrame(X=X, y=y, exposure=exposure, offset=offset)


        def build_model():
            # Example:
            # from superglm import Categorical, SuperGLM
            #
            # return SuperGLM(
            #     family="poisson",
            #     discrete=True,
            #     features={{"rating_factor": Categorical()}},
            # )
            raise NotImplementedError("Configure and return a SuperGLM model")
        """
    )


def _custom_spec_template() -> str:
    return dedent(
        """\
        from __future__ import annotations

        from pathlib import Path

        from pricing_pipeline.models.config import load_model_build_config


        MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))
        """
    )


def _custom_data_template(*, package_name: str) -> str:
    table_prefix = f"{package_name.upper()}_MODEL_READY"
    return dedent(
        f"""\
        from __future__ import annotations

        from pathlib import Path
        from typing import Any, Mapping

        from pricing_models.{package_name}.spec import MODEL_CONFIG
        from pricing_pipeline.models.spec import DatasetSpec
        from pricing_pipeline.orchestration.run_context import scoped_identifier


        DATASET_NAME = "{package_name}_model_ready"
        SOURCE_SYSTEM = "sql_server"
        MODEL_READY_SCHEMA = "REPLACE_ME_SCHEMA"
        MODEL_READY_TABLE_PREFIX = "{table_prefix}"
        PK_COLUMNS = ("REPLACE_ME_ID",)
        WEIGHT_COLUMN: str | None = None
        DEFAULT_OUTPUT_ROOT = Path("state/{package_name}")


        def model_ready_table_for_run(run_key: object | None) -> str:
            return scoped_identifier(MODEL_READY_TABLE_PREFIX, run_key)


        def manifest_sql_for_table(schema_name: str, table_name: str) -> str:
            order_by = ", ".join(PK_COLUMNS)
            return (
                "SELECT *\\n"
                f"FROM {{schema_name}}.{{table_name}}\\n"
                f"ORDER BY {{order_by}}"
            )


        def dataset_spec_from_prepared(prepared: Mapping[str, Any]) -> DatasetSpec:
            schema_name = str(prepared.get("modeling_schema") or MODEL_READY_SCHEMA)
            table_name = str(prepared["modeling_table"])
            return DatasetSpec(
                dataset_name=DATASET_NAME,
                source_system=SOURCE_SYSTEM,
                manifest_sql=manifest_sql_for_table(schema_name, table_name),
                pk_columns=PK_COLUMNS,
                target_column=MODEL_CONFIG.target_name,
                weight_column=WEIGHT_COLUMN,
                raw_loader=None,
            )


        def prepare_model_ready_data(
            engine,
            *,
            run_key: str,
            output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        ) -> dict[str, str]:
            output_dir = Path(output_root) / run_key
            output_dir.mkdir(parents=True, exist_ok=True)
            table_name = model_ready_table_for_run(run_key)

            raise NotImplementedError(
                "Load source data, build final model-ready features, write a "
                f"run-specific table/file for {{table_name}} under {{output_dir}}, "
                "then return modeling_schema, "
                "modeling_table, training_frame_path, output_dir, and run_key."
            )
        """
    )


def _custom_modeling_template(*, package_name: str) -> str:
    return dedent(
        f"""\
        from __future__ import annotations

        from datetime import date, datetime
        from pathlib import Path
        from typing import Any, Mapping

        from sqlalchemy import text

        from pricing_models.{package_name}.spec import MODEL_CONFIG
        from pricing_pipeline.infra.schema import schema_names_from_connectable
        from pricing_pipeline.orchestration.publish_completed_build import (
            CompletedModelBuild,
        )
        from pricing_pipeline.publishing.rating_export import build_export_id


        def effective_from_for_run(value: date | datetime | str | None = None) -> str:
            if value is None:
                return date.today().isoformat()
            if isinstance(value, datetime):
                return value.date().isoformat()
            if isinstance(value, date):
                return value.isoformat()
            cleaned = str(value).strip()
            if not cleaned:
                return date.today().isoformat()
            return cleaned[:10]


        def existing_model_version_for_export(
            engine,
            *,
            model_key: str,
            export_id: str,
        ) -> str | None:
            schemas = schema_names_from_connectable(engine)
            with engine.begin() as con:
                version = con.execute(
                    text(
                        f\"\"\"
                        SELECT TOP (1) rp.model_version
                        FROM {{schemas.pricing}}.PRICING_RATE_PACKAGE AS rp
                        JOIN {{schemas.pricing}}.PRICING_MODEL AS pm
                          ON pm.model_id = rp.model_id
                        WHERE pm.model_key = :model_key
                          AND rp.source_export_id = :export_id
                        ORDER BY rp.rate_package_id DESC
                        \"\"\"
                    ),
                    {{"model_key": model_key, "export_id": export_id}},
                ).scalar()
            return None if version is None else str(version)


        def next_model_version(engine, *, model_key: str) -> str:
            schemas = schema_names_from_connectable(engine)
            with engine.begin() as con:
                versions = list(
                    con.execute(
                        text(
                            f\"\"\"
                            SELECT rp.model_version
                            FROM {{schemas.pricing}}.PRICING_RATE_PACKAGE AS rp
                            JOIN {{schemas.pricing}}.PRICING_MODEL AS pm
                              ON pm.model_id = rp.model_id
                            WHERE pm.model_key = :model_key
                              AND rp.parent_rate_package_id IS NULL
                            \"\"\"
                        ),
                        {{"model_key": model_key}},
                    ).scalars()
                )
            version_numbers = [
                int(str(value).removeprefix("v"))
                for value in versions
                if str(value).startswith("v") and str(value).removeprefix("v").isdigit()
            ]
            return f"v{{max(version_numbers, default=0) + 1}}"


        def resolve_model_version(engine, *, model_key: str, export_id: str) -> str:
            existing = existing_model_version_for_export(
                engine,
                model_key=model_key,
                export_id=export_id,
            )
            if existing is not None:
                return existing
            return next_model_version(engine, model_key=model_key)


        def completed_model_build_payload(
            prepared: Mapping[str, Any],
            *,
            rating_workbook_path: str | Path,
            model_version: str,
            effective_from: str,
            export_id: str,
            created_by: str,
            model_artifact_path: str | Path | None = None,
            metrics: dict[str, float] | None = None,
        ) -> dict[str, Any]:
            return CompletedModelBuild(
                rating_workbook_path=str(rating_workbook_path),
                model_version=model_version,
                effective_from=effective_from,
                created_by=created_by,
                export_id=export_id,
                manifest_id=prepared.get("manifest_id"),
                split_set_id=prepared.get("split_set_id"),
                mlflow_run_id=None,
                model_artifact_path=(
                    str(model_artifact_path) if model_artifact_path is not None else None
                ),
                metrics=metrics or {{}},
            ).to_dict()


        def train_validate_export_model(
            prepared: Mapping[str, Any],
            *,
            engine,
            created_by: str = "airflow",
        ) -> dict[str, Any]:
            run_key = str(prepared.get("run_key") or "manual")
            export_id = build_export_id(MODEL_CONFIG.model_key, run_key)
            model_version = resolve_model_version(
                engine,
                model_key=MODEL_CONFIG.model_key,
                export_id=export_id,
            )
            effective_from = effective_from_for_run(prepared.get("effective_from"))

            raise NotImplementedError(
                "Read the prepared model-ready data, fit/validate SuperGLM, export "
                "the rating workbook, then return completed_model_build_payload(...) "
                f"with export_id={{export_id!r}}, model_version={{model_version!r}}, "
                f"and effective_from={{effective_from!r}}."
            )
        """
    )


def _custom_airflow_tasks_template(*, package_name: str) -> str:
    return dedent(
        f"""\
        from __future__ import annotations

        from pathlib import Path
        from typing import Any, Mapping

        from pricing_models.{package_name}.data import (
            DEFAULT_OUTPUT_ROOT,
            prepare_model_ready_data,
        )
        from pricing_models.{package_name}.modeling import train_validate_export_model
        from pricing_pipeline.orchestration.run_context import run_key_for_value


        def prepare_model_ready_data_task(
            *,
            output_root: str | Path = DEFAULT_OUTPUT_ROOT,
            runtime_module: str | None = None,
            task_id: str = "prepare_model_ready_data",
        ):
            from airflow.sdk import get_current_context, task
            from pricing_pipeline.infra.runtime import runtime_from_env_or_module

            @task(task_id=task_id)
            def _prepare_model_ready_data() -> dict[str, str]:
                runtime = runtime_from_env_or_module(runtime_module)
                context = get_current_context()
                run_value = (
                    context.get("run_id")
                    or _context_logical_date(context)
                    or "manual"
                )
                run_key = run_key_for_value(run_value)
                return prepare_model_ready_data(
                    runtime.get_engine(),
                    run_key=run_key,
                    output_root=output_root,
                )

            return _prepare_model_ready_data


        def train_validate_export_task(
            *,
            runtime_module: str | None = None,
            created_by: str = "airflow",
            task_id: str = "train_validate_export",
        ):
            from airflow.sdk import task
            from pricing_pipeline.infra.runtime import runtime_from_env_or_module

            @task(task_id=task_id)
            def _train_validate_export(prepared: Mapping[str, Any]) -> dict[str, Any]:
                runtime = runtime_from_env_or_module(runtime_module)
                payload = train_validate_export_model(
                    dict(prepared),
                    engine=runtime.get_engine(),
                    created_by=created_by,
                )
                payload.setdefault("manifest_id", prepared.get("manifest_id"))
                payload.setdefault("split_set_id", prepared.get("split_set_id"))
                return payload

            return _train_validate_export


        def _context_logical_date(context: Mapping[str, Any]) -> object | None:
            value = context.get("logical_date")
            if value is not None:
                return value
            dag_run = context.get("dag_run")
            return (
                getattr(dag_run, "logical_date", None)
                or getattr(dag_run, "run_after", None)
                or getattr(dag_run, "execution_date", None)
            )
        """
    )


def _custom_dag_template(*, package_name: str, dag_id: str) -> str:
    tag = package_name.replace("_", "-")
    return dedent(
        f"""\
        from __future__ import annotations

        from airflow.sdk import dag

        from pricing_models.{package_name}.airflow_tasks import (
            prepare_model_ready_data_task,
            train_validate_export_task,
        )
        from pricing_models.{package_name}.data import dataset_spec_from_prepared
        from pricing_models.{package_name}.spec import MODEL_CONFIG
        from pricing_pipeline.orchestration.manifest_tasks import (
            create_prepared_dataset_manifest_task,
        )
        from pricing_pipeline.orchestration.model_registry_tasks import (
            register_pricing_model_task,
        )
        from pricing_pipeline.orchestration.publish_completed_build import (
            publish_completed_model_build_task,
        )


        @dag(
            dag_id="{dag_id}",
            schedule=None,
            catchup=False,
            tags=["pricing", "{tag}"],
        )
        def {dag_id}():
            registered = register_pricing_model_task(model_config=MODEL_CONFIG)()
            prepared = prepare_model_ready_data_task()()
            manifested = create_prepared_dataset_manifest_task(
                model_config=MODEL_CONFIG,
                dataset_builder=dataset_spec_from_prepared,
            )(prepared)
            completed = train_validate_export_task()(manifested)
            published = publish_completed_model_build_task(
                model_config=MODEL_CONFIG,
            )(completed)

            registered >> prepared >> manifested >> completed >> published


        {dag_id}()
        """
    )


def _factory_spec_template(
    *,
    package_name: str,
    experiment_name: str,
) -> str:
    return dedent(
        f"""\
        from __future__ import annotations

        from pathlib import Path

        from pricing_pipeline.models.config import load_model_build_config
        from pricing_pipeline.models.spec import DatasetSpec, ModelSpec
        from pricing_models.{package_name}.training import (
            FEATURE_COLUMNS,
            TRAINING_SQL,
            build_model,
            build_training_frame,
        )


        MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))

        DATASET_SPEC = DatasetSpec(
            dataset_name="{package_name}_training",
            source_system="sql_server",
            manifest_sql=TRAINING_SQL,
            pk_columns=("REPLACE_ME_ID",),
            target_column=MODEL_CONFIG.target_name,
            weight_column=None,
            raw_loader=None,
        )

        MODEL_SPEC = ModelSpec(
            model_key=MODEL_CONFIG.model_key,
            model_label=MODEL_CONFIG.model_label,
            target_name=MODEL_CONFIG.target_name,
            model_type=MODEL_CONFIG.model_type,
            experiment_name="{experiment_name}",
            deployment_slot=MODEL_CONFIG.deployment_slot,
            dataset=DATASET_SPEC,
            training_sql=TRAINING_SQL,
            feature_columns=tuple(FEATURE_COLUMNS),
            build_model=build_model,
            build_training_frame=build_training_frame,
            package_status=MODEL_CONFIG.default_package_status,
        )
        """
    )


def _factory_dag_template(*, package_name: str, dag_id: str) -> str:
    tag = package_name.replace("_", "-")
    return dedent(
        f"""\
        from __future__ import annotations

        from pricing_models.{package_name}.spec import MODEL_CONFIG, MODEL_SPEC
        from pricing_pipeline.orchestration.dag_factory import build_pricing_model_dag


        {dag_id} = build_pricing_model_dag(
            dag_id="{dag_id}",
            spec=MODEL_SPEC,
            model_config=MODEL_CONFIG,
            tags=["pricing", "{tag}"],
        )
        """
    )


def scaffold_pricing_model(options: ScaffoldOptions) -> ScaffoldResult:
    model_key = _validate_model_key(options.model_key)
    template = _required_text(options.template, "template")
    if template not in {"custom", "factory"}:
        raise ValueError("template must be one of: custom, factory")
    package_name = _validate_python_identifier(
        options.package_name or _module_name_from_model_key(model_key),
        "package_name",
    )
    dag_id = _validate_python_identifier(
        options.dag_id or f"pricing_{package_name}",
        "dag_id",
    )
    target_name = _required_text(options.target_name, "target_name")
    model_label = _required_text(
        options.model_label or _default_label(model_key),
        "model_label",
    )
    model_type = _required_text(options.model_type, "model_type")
    deployment_slot = _required_text(
        options.deployment_slot or f"{model_key}_UAT",
        "deployment_slot",
    )
    experiment_name = _required_text(
        options.experiment_name or f"pricing-{package_name.replace('_', '-')}",
        "experiment_name",
    )

    root = options.root
    package_dir = root / "pricing_models" / package_name
    dag_path = root / "dags" / f"{dag_id}.py"
    if template == "factory":
        files = (
            package_dir / "__init__.py",
            package_dir / "model.toml",
            package_dir / "training.py",
            package_dir / "spec.py",
            dag_path,
        )
    else:
        files = (
            package_dir / "__init__.py",
            package_dir / "model.toml",
            package_dir / "spec.py",
            package_dir / "data.py",
            package_dir / "modeling.py",
            package_dir / "airflow_tasks.py",
            dag_path,
        )

    existing = [path for path in files if path.exists()]
    if existing and not options.force:
        existing_text = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {existing_text}")

    package_dir.mkdir(parents=True, exist_ok=True)
    dag_path.parent.mkdir(parents=True, exist_ok=True)

    content_by_path = {
        package_dir / "__init__.py": "",
        package_dir / "model.toml": _model_toml_template(
            model_key=model_key,
            model_label=model_label,
            target_name=target_name,
            model_type=model_type,
            deployment_slot=deployment_slot,
        ),
    }
    if template == "factory":
        content_by_path.update(
            {
                package_dir / "training.py": _training_template(target_name=target_name),
                package_dir / "spec.py": _factory_spec_template(
                    package_name=package_name,
                    experiment_name=experiment_name,
                ),
                dag_path: _factory_dag_template(package_name=package_name, dag_id=dag_id),
            }
        )
    else:
        content_by_path.update(
            {
                package_dir / "spec.py": _custom_spec_template(),
                package_dir / "data.py": _custom_data_template(package_name=package_name),
                package_dir / "modeling.py": _custom_modeling_template(
                    package_name=package_name,
                ),
                package_dir / "airflow_tasks.py": _custom_airflow_tasks_template(
                    package_name=package_name,
                ),
                dag_path: _custom_dag_template(package_name=package_name, dag_id=dag_id),
            }
        )
    for path in files:
        path.write_text(content_by_path[path], encoding="utf-8")

    return ScaffoldResult(
        package_name=package_name,
        dag_id=dag_id,
        created_files=files,
    )


def parse_args(argv: list[str] | None = None) -> ScaffoldOptions:
    parser = argparse.ArgumentParser(
        description="Create a pricing model package and Airflow DAG scaffold.",
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--model-type", default="superglm_poisson")
    parser.add_argument("--deployment-slot")
    parser.add_argument("--package-name")
    parser.add_argument("--dag-id")
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--template",
        choices=("custom", "factory"),
        default="custom",
        help=(
            "custom creates the user-owned TaskFlow DAG scaffold; factory creates "
            "the older build_pricing_model_dag scaffold"
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return ScaffoldOptions(
        model_key=args.model_key,
        model_label=args.model_label,
        target_name=args.target_name,
        model_type=args.model_type,
        deployment_slot=args.deployment_slot,
        package_name=args.package_name,
        dag_id=args.dag_id,
        experiment_name=args.experiment_name,
        template=args.template,
        root=args.root,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> None:
    result = scaffold_pricing_model(parse_args(argv))

    print("created pricing model scaffold:")
    for path in result.created_files:
        print(f"  {path.as_posix()}")
    print()
    print(f"model is auto-discovered from pricing_models/{result.package_name}/model.toml")


if __name__ == "__main__":
    main()
