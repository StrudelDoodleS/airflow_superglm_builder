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
        # Model housekeeping config. Keep model identity, deployment lane, and
        # validation split settings here; keep source SQL and Python model code
        # in the neighboring package files.
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


def _package_init_template(*, package_name: str) -> str:
    return f'"""Pricing model package for {package_name}."""\n'


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
        \"\"\"Load the TOML model configuration for this pricing model.\"\"\"

        from __future__ import annotations

        from pathlib import Path

        from pricing_pipeline.models.config import load_model_build_config


        MODEL_CONFIG = load_model_build_config(Path(__file__).with_name("model.toml"))
        """
    )


def _custom_source_sql_template(*, model_key: str) -> str:
    return dedent(
        f"""\
        -- Source-data query placeholder for {model_key}.
        --
        -- Put model-local source SQL here when that is convenient, then read it
        -- from data.py. You can also ignore this file and call your team's
        -- existing Python data-access helper from data.py instead.
        --
        -- Keep this query focused on source/prepared data. Target construction,
        -- feature engineering, filtering, and final feature selection can still
        -- happen in modeling.py.
        --
        -- Example:
        -- SELECT *
        -- FROM some_schema.some_source_view;
        """
    )


def _custom_data_template(*, package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Read or stage source data for this pricing model.

        This file is model-owned. Define how the DAG obtains source data here:
        read SQL from sql/source_data.sql, call your team's existing connection
        helper, copy a run-scoped extract, or delegate to another local module.
        Return only small run metadata for downstream modeling tasks.
        \"\"\"

        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        DATASET_NAME = "{package_name}_model_frame"
        SOURCE_SYSTEM = "sql_server"
        PK_COLUMNS = ("REPLACE_ME_ID",)
        WEIGHT_COLUMN: str | None = None
        DEFAULT_OUTPUT_ROOT = Path("state/{package_name}")
        SQL_DIR = Path(__file__).with_name("sql")
        SOURCE_DATA_SQL = SQL_DIR / "source_data.sql"


        def prepare_source_data(
            engine,
            *,
            run_key: str,
            output_dir: str | Path,
        ) -> dict[str, Any]:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            raise NotImplementedError(
                "Read or stage source data for this run, write any temporary "
                f"artifacts under {{output_path}}, then return output_dir, "
                "effective_from, data_as_of_date, and any paths/IDs needed by "
                "modeling.py."
            )
        """
    )


def _custom_modeling_template(*, package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Model-owned build logic for this pricing model.

        Edit the functions in the first section. The final recipe function wires
        those pieces into the shared manifest/publish contract.
        \"\"\"

        from __future__ import annotations

        from pathlib import Path
        from typing import Any, Mapping

        import pandas as pd

        # Model-local config/constants.
        from pricing_models.{package_name}.data import (
            DATASET_NAME,
            PK_COLUMNS,
            SOURCE_SYSTEM,
            WEIGHT_COLUMN,
        )
        from pricing_models.{package_name}.spec import MODEL_CONFIG

        # Shared lifecycle helpers. Most model authors do not need to edit these imports.
        from pricing_pipeline.data.manifest import (
            ModelFrameManifestSpec,
            create_model_frame_manifest_with_split,
            validation_split_indices,
        )
        from pricing_pipeline.orchestration.completed_build_helpers import (
            completed_model_build_payload,
            effective_from_for_run,
            required_payload_text,
        )
        from pricing_pipeline.publishing.model_versions import (
            resolve_model_version_for_export,
        )
        from pricing_pipeline.publishing.rating_export import build_export_id


        # ---------------------------------------------------------------------------
        # Edit These Model-Specific Functions
        # ---------------------------------------------------------------------------

        def read_prepared_source(prepared: Mapping[str, Any]) -> pd.DataFrame:
            \"\"\"Load the prepared source frame for this run.

            data.py decides the handoff shape. For example, if prepare_source_data(...)
            returned {{"source_data_path": ".../source.parquet"}}, read that file here.
            Do not pass large DataFrames through Airflow/XCom.
            \"\"\"
            raise NotImplementedError(
                "Read the source artifact/table identified by prepared and return a "
                "pandas DataFrame."
            )


        def build_final_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
            \"\"\"Create the final frame used for validation, training, export, and manifesting.\"\"\"
            # Add target construction, pd.cut/binning, feature engineering, filtering,
            # and final feature selection here.
            return raw.copy()


        def fit_validate_export_rating_tables(
            frame: pd.DataFrame,
            *,
            split_indices: list[tuple[Any, Any]],
            output_dir: str | Path,
            model_version: str,
            effective_from: str,
        ) -> tuple[str | Path, str | Path | None, dict[str, float]]:
            \"\"\"Fit/validate the model and export the rating workbook.

            Return:
                rating_workbook_path, model_artifact_path, metrics
            \"\"\"
            raise NotImplementedError(
                "Fit/validate the model using split_indices, export the rating "
                "workbook, optionally persist the model artifact, and return "
                "(rating_workbook_path, model_artifact_path, metrics)."
            )


        def validation_split_indices_for_model(
            frame: pd.DataFrame,
        ) -> list[tuple[Any, Any]]:
            \"\"\"Return the validation folds used by this model.

            Built-in model.toml methods delegate to pricing_pipeline. If model.toml
            uses method = "custom", replace this function body with model-specific
            positional row indices, for example from a SQL lookup, external mapping,
            grouping rule, or temporal rule.
            \"\"\"
            if MODEL_CONFIG.validation_split.method == "custom":
                raise NotImplementedError(
                    "Return custom validation folds as "
                    "[(train_idx, test_idx), ...] using zero-based row positions."
                )
            return validation_split_indices(frame, MODEL_CONFIG.validation_split)


        # ---------------------------------------------------------------------------
        # Standard Build Recipe - Usually Leave This Alone
        # ---------------------------------------------------------------------------

        def train_validate_export_model(
            prepared: Mapping[str, Any],
            *,
            engine,
            settings,
            created_by: str = "airflow",
        ) -> dict[str, Any]:
            \"\"\"Standard custom-model lifecycle recipe.

            Start by customizing the functions above. Edit this recipe only when your
            model needs a different build flow. The recipe resolves stable publish
            metadata, calls the model-owned functions, creates the frame-backed
            manifest, and returns the CompletedModelBuild payload consumed by the
            publish task.
            \"\"\"
            run_key = str(prepared.get("run_key") or "manual")
            export_id = build_export_id(MODEL_CONFIG.model_key, run_key)
            model_version = resolve_model_version_for_export(
                engine,
                model_key=MODEL_CONFIG.model_key,
                export_id=export_id,
            )
            effective_from = effective_from_for_run(
                required_payload_text(prepared, "effective_from")
            )
            data_as_of_date = required_payload_text(prepared, "data_as_of_date")

            raw = read_prepared_source(prepared)
            frame = build_final_model_frame(raw)
            # The manifest and split artifacts use this frame order, so keep ordering
            # deterministic and aligned with PK_COLUMNS unless the model deliberately needs
            # a different order.
            frame = frame.sort_values(list(PK_COLUMNS)).reset_index(drop=True)
            split_indices = validation_split_indices_for_model(frame)
            # If validation_split uses a source split column, do not include that
            # column as a rating feature unless this is an intentional model decision.
            rating_workbook_path, model_artifact_path, metrics = (
                fit_validate_export_rating_tables(
                    frame,
                    split_indices=split_indices,
                    output_dir=prepared.get("output_dir") or Path("state") / run_key,
                    model_version=model_version,
                    effective_from=effective_from,
                )
            )
            manifest = create_model_frame_manifest_with_split(
                engine,
                frame=frame,
                spec=ModelFrameManifestSpec(
                    dataset_name=DATASET_NAME,
                    source_system=SOURCE_SYSTEM,
                    data_as_of_date=data_as_of_date,
                    pk_columns=PK_COLUMNS,
                    target_column=MODEL_CONFIG.target_name,
                    weight_column=WEIGHT_COLUMN,
                ),
                validation_split=MODEL_CONFIG.validation_split,
                validation_split_artifact_root=settings.validation_split_artifact_root,
                split_indices=split_indices,
                created_by=created_by,
            )

            return completed_model_build_payload(
                rating_workbook_path=rating_workbook_path,
                model_version=model_version,
                effective_from=effective_from,
                export_id=export_id,
                created_by=created_by,
                manifest_id=manifest.manifest_id,
                split_set_id=manifest.split_set_id,
                model_artifact_path=model_artifact_path,
                metrics=metrics,
            )
        """
    )


def _custom_airflow_tasks_template(*, package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Airflow TaskFlow wrappers for this model package.

        Keep business logic in data.py and modeling.py. This file should stay
        thin: load runtime config, attach @task decorators, and pass small
        dictionaries between Airflow tasks.
        \"\"\"

        from __future__ import annotations

        from pathlib import Path
        from typing import Any, Mapping

        from pricing_models.{package_name}.data import (
            DEFAULT_OUTPUT_ROOT,
            prepare_source_data,
        )
        from pricing_models.{package_name}.modeling import (
            train_validate_export_model,
        )
        from pricing_pipeline.orchestration.airflow_run_metadata import (
            merge_prepared_payload_metadata,
            task_run_metadata,
        )


        def prepare_source_data_task(
            *,
            output_root: str | Path = DEFAULT_OUTPUT_ROOT,
            runtime_module: str | None = None,
            task_id: str = "prepare_source_data",
        ):
            from airflow.sdk import get_current_context, task
            from pricing_pipeline.infra.runtime import runtime_from_env_or_module

            @task(task_id=task_id)
            def _prepare_source_data() -> dict[str, Any]:
                runtime = runtime_from_env_or_module(runtime_module)
                metadata = task_run_metadata(
                    get_current_context(),
                    output_root=output_root,
                )
                payload = prepare_source_data(
                    runtime.get_engine(),
                    run_key=metadata["run_key"],
                    output_dir=Path(metadata["output_dir"]),
                )
                return merge_prepared_payload_metadata(metadata, payload)

            return _prepare_source_data


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
                    settings=runtime.settings,
                    created_by=created_by,
                )
                return payload

            return _train_validate_export

        """
    )


def _custom_dag_template(*, package_name: str, dag_id: str, model_key: str) -> str:
    tag = package_name.replace("_", "-")
    return dedent(
        f"""\
        \"\"\"Airflow DAG for the {model_key} pricing model build.\"\"\"

        from __future__ import annotations

        from airflow.sdk import dag

        from pricing_models.{package_name}.airflow_tasks import (
            prepare_source_data_task,
            train_validate_export_task,
        )
        from pricing_models.{package_name}.spec import MODEL_CONFIG
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
            prepared = prepare_source_data_task()()
            completed = train_validate_export_task()(prepared)
            published = publish_completed_model_build_task(
                model_config=MODEL_CONFIG,
            )(completed)

            registered >> prepared >> completed >> published


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
            package_dir / "sql" / "source_data.sql",
            package_dir / "spec.py",
            package_dir / "data.py",
            package_dir / "modeling.py",
            package_dir / "airflow_tasks.py",
            dag_path,
        )

    package_dir.mkdir(parents=True, exist_ok=True)
    dag_path.parent.mkdir(parents=True, exist_ok=True)

    content_by_path = {
        package_dir / "__init__.py": _package_init_template(package_name=package_name),
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
                package_dir / "sql" / "source_data.sql": _custom_source_sql_template(
                    model_key=model_key
                ),
                package_dir / "spec.py": _custom_spec_template(),
                package_dir / "data.py": _custom_data_template(package_name=package_name),
                package_dir / "modeling.py": _custom_modeling_template(
                    package_name=package_name,
                ),
                package_dir / "airflow_tasks.py": _custom_airflow_tasks_template(
                    package_name=package_name,
                ),
                dag_path: _custom_dag_template(
                    package_name=package_name,
                    dag_id=dag_id,
                    model_key=model_key,
                ),
            }
        )
    created_files: list[Path] = []
    for path in files:
        if path.exists() and not options.force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content_by_path[path], encoding="utf-8")
        created_files.append(path)

    return ScaffoldResult(
        package_name=package_name,
        dag_id=dag_id,
        created_files=tuple(created_files),
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
