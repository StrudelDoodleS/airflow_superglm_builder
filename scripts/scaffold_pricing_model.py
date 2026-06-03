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
    root: Path = Path(".")
    force: bool = False


@dataclass(frozen=True)
class ScaffoldResult:
    package_name: str
    dag_id: str
    created_files: tuple[Path, ...]
    registry_instructions: tuple[str, ...]


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
            "model_key must start with a letter and contain only letters, numbers, "
            "and underscores"
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


def _spec_template(
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


def _dag_template(*, package_name: str, dag_id: str) -> str:
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


def _registry_instructions(*, package_name: str, model_key: str) -> tuple[str, ...]:
    return (
        f"from pricing_models.{package_name}.spec import MODEL_CONFIG as {model_key}_CONFIG",
        f"from pricing_models.{package_name}.spec import MODEL_SPEC as {model_key}_SPEC",
        f"MODEL_SPECS[{model_key}_SPEC.model_key] = {model_key}_SPEC",
        f"MODEL_CONFIGS[{model_key}_CONFIG.model_key] = {model_key}_CONFIG",
    )


def scaffold_pricing_model(options: ScaffoldOptions) -> ScaffoldResult:
    model_key = _validate_model_key(options.model_key)
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
    files = (
        package_dir / "__init__.py",
        package_dir / "model.toml",
        package_dir / "training.py",
        package_dir / "spec.py",
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
        package_dir / "training.py": _training_template(target_name=target_name),
        package_dir / "spec.py": _spec_template(
            package_name=package_name,
            experiment_name=experiment_name,
        ),
        dag_path: _dag_template(package_name=package_name, dag_id=dag_id),
    }
    for path in files:
        path.write_text(content_by_path[path], encoding="utf-8")

    return ScaffoldResult(
        package_name=package_name,
        dag_id=dag_id,
        created_files=files,
        registry_instructions=_registry_instructions(
            package_name=package_name,
            model_key=model_key,
        ),
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
        root=args.root,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> None:
    result = scaffold_pricing_model(parse_args(argv))

    print("created pricing model scaffold:")
    for path in result.created_files:
        print(f"  {path.as_posix()}")
    print()
    print("add these lines to pricing_models/registry.py:")
    for line in result.registry_instructions:
        print(f"  {line}")


if __name__ == "__main__":
    main()
