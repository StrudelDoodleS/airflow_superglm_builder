from __future__ import annotations

import argparse
import json
import keyword
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DOTTED_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_TEMPLATE_TOKEN = re.compile(r"__[A-Z][A-Z0-9_]*__")
_DEFAULT_CONFIG_NAME = "pricing_scaffold.toml"
_CONFIG_SECTION = "notebook_defaults"
_CONFIG_KEYS = frozenset(
    {
        "database_mode",
        "runtime_module",
        "expected_remote_database",
    }
)
_NOTEBOOK_NAMES = (
    "01_data_ingestion.ipynb",
    "02_model_training.ipynb",
    "03_model_editor.ipynb",
    "04_model_deployment.ipynb",
    "99_scratch_work.ipynb",
)


@dataclass(frozen=True)
class ScaffoldOptions:
    model_name: str
    target_name: str
    model_label: str | None = None
    model_type: str = "superglm_poisson"
    deployment_slot: str | None = None
    package_name: str | None = None
    database_mode: str = "local"
    runtime_module: str | None = None
    expected_remote_database: str = ""
    root: Path = Path(".")
    force: bool = False


@dataclass(frozen=True)
class ScaffoldResult:
    package_name: str
    created_files: tuple[Path, ...]


@dataclass(frozen=True)
class ScaffoldConfig:
    database_mode: str = "local"
    runtime_module: str | None = None
    expected_remote_database: str = ""


def _required(value: str | None, name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{name} is required")
    return cleaned


def _model_name(value: str) -> str:
    cleaned = _required(value, "model_name")
    if not _MODEL_NAME.fullmatch(cleaned):
        raise ValueError(
            "model_name must start with a letter and contain only letters, numbers, and underscores"
        )
    return cleaned


def _package_name(value: str) -> str:
    cleaned = _required(value, "package_name")
    if not _PYTHON_IDENTIFIER.fullmatch(cleaned) or keyword.iskeyword(cleaned):
        raise ValueError("package_name must be a valid Python identifier")
    return cleaned


def _database_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("database_mode must be 'local' or 'remote'")
    cleaned = value.strip().lower()
    if cleaned not in {"local", "remote"}:
        raise ValueError("database_mode must be 'local' or 'remote'")
    return cleaned


def _runtime_module(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("runtime_module must be a dotted Python module name or an empty string")
    cleaned = value.strip()
    if not cleaned:
        return None
    if not _DOTTED_MODULE.fullmatch(cleaned):
        raise ValueError("runtime_module must be a dotted Python module name")
    return cleaned


def _expected_remote_database(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected_remote_database must be a string")
    return value.strip()


def load_scaffold_config(path: str | Path) -> ScaffoldConfig:
    """Load strict, non-secret notebook connection defaults from TOML."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"scaffold config does not exist: {config_path}")
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"scaffold config could not be read: {config_path}: {exc}") from exc
    unexpected_sections = sorted(set(payload) - {_CONFIG_SECTION})
    if unexpected_sections:
        raise ValueError(
            "scaffold config has unsupported top-level sections: " + ", ".join(unexpected_sections)
        )
    raw = payload.get(_CONFIG_SECTION, {})
    if not isinstance(raw, dict):
        raise TypeError(f"[{_CONFIG_SECTION}] must be a TOML table")
    unexpected_keys = sorted(set(raw) - _CONFIG_KEYS)
    if unexpected_keys:
        raise ValueError(f"[{_CONFIG_SECTION}] has unsupported keys: " + ", ".join(unexpected_keys))
    return ScaffoldConfig(
        database_mode=_database_mode(raw.get("database_mode", "local")),
        runtime_module=_runtime_module(raw.get("runtime_module")),
        expected_remote_database=_expected_remote_database(raw.get("expected_remote_database", "")),
    )


def _python_literal(value: object) -> str:
    if value is None:
        return "None"
    return json.dumps(value, ensure_ascii=False)


def _render_template(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, str):
        return _TEMPLATE_TOKEN.sub(lambda match: replacements[match.group()], value)
    if isinstance(value, list):
        return [_render_template(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _render_template(item, replacements) for key, item in value.items()}
    return value


def _markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def _notebook_document(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _project_setup_source(*, imports: str) -> str:
    setup = dedent(
        """
        from pathlib import Path
        import sys

        search_root = Path.cwd().resolve()
        PROJECT_ROOT = next(
            (
                root
                for root in (search_root, *search_root.parents)
                if (root / "pricing_pipeline").is_dir()
                and (root / "pricing_models").is_dir()
            ),
            None,
        )
        if PROJECT_ROOT is None:
            raise RuntimeError("Open this notebook from inside the pricing repository.")
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        """
    ).strip()
    paths = dedent(
        """
        MODEL_DIR = PROJECT_ROOT / "pricing_models/__PACKAGE_NAME__"
        FRAME_ARTIFACT_PATH = MODEL_DIR / ".local" / "model_frame.joblib"
        GROUPING_ARTIFACT_PATH = MODEL_DIR / ".local" / "routine_groupings.joblib"
        """
    ).strip()
    return f"{setup}\n\n{dedent(imports).strip()}\n\n{paths}\n"


def _template_notebooks(
    *,
    database_mode: str,
    runtime_module: str | None,
    expected_remote_database: str,
) -> dict[str, dict[str, object]]:
    connection_settings = (
        dedent(
            f"""
            DATABASE_MODE = {_python_literal(database_mode)}  # "local" or "remote"
            RUNTIME_MODULE = {_python_literal(runtime_module)}  # e.g. "work_runtime.database"; never put secrets here
            EXPECTED_REMOTE_DATABASE = {_python_literal(expected_remote_database)}
            ALLOW_REMOTE_WRITES = False
            """
        ).strip()
        + "\n"
    )
    connect_source = """
    pricing = connect(
        mode=DATABASE_MODE,
        runtime_module=RUNTIME_MODULE,
        local_root=MODEL_DIR / ".local",
        expected_remote_database=EXPECTED_REMOTE_DATABASE,
        allow_remote_writes=ALLOW_REMOTE_WRITES,
    )
    display(pricing.destination)
    """

    ingestion = _notebook_document(
        [
            _markdown(
                """
                # __MODEL_LABEL_MARKDOWN__: data ingestion

                This notebook is the only governed step that creates or replaces the
                final model-frame handoff. Keep the source query and every transform
                visible, then save the verified artifact for model training.
                """
            ),
            _code(
                connection_settings
                + """
DATA_AS_OF = ""  # Required ISO date: the dataset version, not a deployment date.
REPLACE_MODEL_FRAME = False
"""
            ),
            _code(
                _project_setup_source(
                    imports="""
                    import numpy as np  # noqa: E402
                    import pandas as pd  # noqa: E402

                    from pricing_pipeline.notebook import connect, save_model_frame  # noqa: E402
                    """,
                )
            ),
            _markdown("## Connect and verify the source/audit destination"),
            _code(connect_source),
            _markdown(
                """
                ## Read source data

                Replace this demo with the normal work query. Keep source-owned keys,
                target, exposure/weights, split fields, and the data-as-of value.
                """
            ),
            _code(
                """
                if not DATA_AS_OF.strip():
                    raise ValueError("Set the required DATA_AS_OF dataset version stamp.")
                rng = np.random.default_rng(42)
                raw = pd.DataFrame({
                    "__PRIMARY_KEY__": np.arange(1, 101),
                    "__FEATURE_NAME__": rng.normal(size=100),
                    "segment": rng.choice(["A", "B", "C"], size=100),
                    "data_as_of": [DATA_AS_OF] * 100,
                })
                raw["__TARGET_NAME__"] = rng.poisson(
                    np.exp(
                        -0.5
                        + 0.25 * raw["__FEATURE_NAME__"]
                        + raw["segment"].map({"A": 0.0, "B": 0.2, "C": -0.1})
                    )
                )
                display({"Rows loaded": len(raw), "Columns loaded": len(raw.columns)})
                """
            ),
            _markdown(
                """
                ## Build the final model frame

                Apply production-intended feature transforms here. Preserve deterministic
                row ordering and retain only the columns needed by the training spec.
                """
            ),
            _code(
                """
                frame = (
                    raw.loc[
                        :,
                        [
                            "__PRIMARY_KEY__",
                            "__TARGET_NAME__",
                            "__FEATURE_NAME__",
                            "segment",
                            "data_as_of",
                        ],
                    ]
                    .sort_values("__PRIMARY_KEY__")
                    .reset_index(drop=True)
                )
                display(frame.head())
                """
            ),
            _markdown("## Save the verified notebook handoff"),
            _code(
                """
                frame_artifact = save_model_frame(
                    frame,
                    FRAME_ARTIFACT_PATH,
                    replace=REPLACE_MODEL_FRAME,
                )
                display(frame_artifact)
                """
            ),
        ]
    )

    training = _notebook_document(
        [
            _markdown(
                """
                # __MODEL_LABEL_MARKDOWN__: model training

                Fit and publish the untouched raw model first. The later routine-edit
                section is optional and contains the one visible place for pre-applied
                level collapses or other standard simplifications. No live editor runs here.
                """
            ),
            _code(connection_settings),
            _code(
                _project_setup_source(
                    imports="""
                    from superglm import Categorical, Numeric, SuperGLM  # noqa: E402

                    from pricing_pipeline.models.config import ValidationSplitConfig  # noqa: E402
                    from pricing_pipeline.notebook import (  # noqa: E402
                        PricingModelSpec,
                        apply_level_groupings,
                        build_candidate,
                        connect,
                        inspect_level_groupings,
                        load_level_groupings,
                        load_model_frame,
                        publish_candidate,
                        register_model,
                    )
                    """,
                )
            ),
            _markdown("## Connect and load the exact ingested frame"),
            _code(connect_source),
            _code(
                """
                frame = load_model_frame(FRAME_ARTIFACT_PATH)
                display({"Rows": len(frame), "Columns": len(frame.columns)})
                """
            ),
            _markdown("## Stable model identity and validation decision"),
            _code(
                """
                MODEL = PricingModelSpec(
                    name="__MODEL_NAME__",
                    label="__MODEL_LABEL__",
                    target="__TARGET_NAME__",
                    model_type="__MODEL_TYPE__",
                    deployment_slot="__DEPLOYMENT_SLOT__",
                    features=("__FEATURE_NAME__", "segment"),
                    dataset_name="__DATASET_NAME__",
                    source_system="replace_with_source_name",
                    pk_columns=("__PRIMARY_KEY__",),
                    offset_column=None,
                    offset_source_column=None,
                    offset_label=None,
                    sample_weight_column=None,
                    export_weight_column=None,
                    data_as_of_column="data_as_of",
                    validation=ValidationSplitConfig.kfold(
                        n_splits=5,
                        random_state=42,
                        shuffle=True,
                    ),
                )
                model = register_model(pricing, MODEL, source_root=MODEL_DIR)
                """
            ),
            _markdown("## Raw model: no pre-applied groupings and no editor"),
            _code(
                """
                RAW_FEATURES = {
                    "__FEATURE_NAME__": Numeric(),
                    "segment": Categorical(),
                }
                raw_superglm_model = SuperGLM(
                    family="poisson",
                    selection_penalty=0.0,
                    discrete=True,
                    n_bins=64,
                    features=RAW_FEATURES,
                )
                """
            ),
            _code(
                """
                raw_candidate = build_candidate(
                    pricing,
                    model=model,
                    frame=frame,
                    superglm_model=raw_superglm_model,
                    model_kind="RAW",
                )
                raw_candidate.metrics
                """
            ),
            _code(
                """
                raw_published = publish_candidate(pricing, raw_candidate)
                display({
                    "Model": raw_published.model_name,
                    "Kind": raw_published.model_kind,
                    "Package": raw_published.package_version,
                    "Manifest": raw_published.manifest_id,
                    "State": raw_published.package_status,
                    "Reused equivalent": raw_published.deduplicated,
                })
                """
            ),
            _markdown(
                """
                ## Optional routine edit: pre-applied simplifications

                `99_scratch_work.ipynb` can export all interactive categorical
                collapses from a reviewed RAW candidate. This cell loads the actual
                `LevelGrouping` objects and automatically skips the routine model when
                the artifact is absent or contains no real collapse.
                """
            ),
            _code(
                """
                LEVEL_GROUPINGS = (
                    load_level_groupings(
                        GROUPING_ARTIFACT_PATH,
                        frame=frame,
                        model=model,
                    )
                    if GROUPING_ARTIFACT_PATH.is_file()
                    else {}
                )
                ROUTINE_EDIT_CONFIGURED = bool(LEVEL_GROUPINGS)
                routine_superglm_model = None
                if ROUTINE_EDIT_CONFIGURED:
                    ROUTINE_FEATURES = apply_level_groupings(RAW_FEATURES, LEVEL_GROUPINGS)
                    routine_superglm_model = SuperGLM(
                        family="poisson",
                        selection_penalty=0.0,
                        discrete=True,
                        n_bins=64,
                        features=ROUTINE_FEATURES,
                    )
                    display(inspect_level_groupings(GROUPING_ARTIFACT_PATH))
                else:
                    display("No exported level collapses: ROUTINE_EDIT skipped.")
                """
            ),
            _code(
                """
                routine_published = None
                if ROUTINE_EDIT_CONFIGURED:
                    routine_candidate = build_candidate(
                        pricing,
                        model=model,
                        frame=frame,
                        superglm_model=routine_superglm_model,
                        model_kind="ROUTINE_EDIT",
                    )
                    display(routine_candidate.metrics)
                    routine_published = publish_candidate(pricing, routine_candidate)
                    display({
                        "Kind": routine_published.model_kind,
                        "Package": routine_published.package_version,
                        "Manifest": routine_published.manifest_id,
                        "State": routine_published.package_status,
                        "Reused equivalent": routine_published.deduplicated,
                    })
                """
            ),
        ]
    )

    editor = _notebook_document(
        [
            _markdown(
                """
                # __MODEL_LABEL_MARKDOWN__: optional editor

                Select any published candidate version for this SQL model, or leave the
                package unset to open the latest. Publishing an editor session creates an
                immutable `EDITOR_EDIT` child package.
                """
            ),
            _code(
                connection_settings
                + """
MODEL_NAME = "__MODEL_NAME__"  # Set to None to select by label only.
MODEL_LABEL = "__MODEL_LABEL__"
DEPLOYMENT_SLOT = "__DEPLOYMENT_SLOT__"
PACKAGE_VERSION = None  # None selects the latest listed package.
EDIT_REASON = ""
"""
            ),
            _code(
                _project_setup_source(
                    imports="""
                    from superglm.editor import EditorSession  # noqa: E402

                    from pricing_pipeline.notebook import (  # noqa: E402
                        connect,
                        list_candidate_versions,
                        load_registered_model,
                        open_candidate,
                        publish_edits,
                    )
                    """,
                )
            ),
            _markdown("## Connect, resolve the model label, and list versions"),
            _code(connect_source),
            _code(
                """
                model = load_registered_model(
                    pricing,
                    model_name=MODEL_NAME,
                    model_label=MODEL_LABEL,
                    deployment_slot=DEPLOYMENT_SLOT,
                    source_root=MODEL_DIR,
                )
                versions = list_candidate_versions(pricing, model=model)
                display(versions)
                """
            ),
            _markdown("## Select and open the exact candidate"),
            _code(
                """
                if versions.empty:
                    raise LookupError("No candidate package versions were found.")
                selected_package_version = (
                    int(versions.iloc[0]["Package"])
                    if PACKAGE_VERSION is None
                    else int(PACKAGE_VERSION)
                )
                if selected_package_version not in set(versions["Package"].astype(int)):
                    raise ValueError("PACKAGE_VERSION is not in the displayed candidate list.")
                reviewed = open_candidate(
                    pricing,
                    model=model,
                    package_version=selected_package_version,
                )
                display(reviewed.technical)
                """
            ),
            _markdown("## Open the live editor"),
            _code(
                """
                editor_session = EditorSession.from_model(
                    reviewed.bundle.fitted_model,
                    train_data=(
                        reviewed.bundle.X,
                        reviewed.bundle.y,
                        reviewed.bundle.sample_weight,
                        reviewed.bundle.offset,
                    ),
                    cv_report=reviewed.bundle.cv_report,
                )
                display(editor_session.widget())
                """
            ),
            _markdown("## Preview the edited model without publishing"),
            _code(
                """
                edited_model = editor_session.to_model()
                edited_model
                """
            ),
            _markdown("## Publish the retained editor session"),
            _code(
                """
                if not EDIT_REASON.strip():
                    raise ValueError("Describe the market or underwriting edit.")
                edited = publish_edits(
                    pricing,
                    candidate=reviewed,
                    editor_session=editor_session,
                    reason=EDIT_REASON,
                )
                display({
                    "Kind": edited.model_kind,
                    "Package": edited.package_version,
                    "Parent package ID": edited.parent_rate_package_id,
                    "State": edited.package_status,
                    "Reused equivalent": edited.deduplicated,
                })
                """
            ),
        ]
    )

    deployment = _notebook_document(
        [
            _markdown(
                """
                # __MODEL_LABEL_MARKDOWN__: deployment

                Select a published SQL candidate, open that exact immutable package for
                review, and explicitly deploy it. The champion snapshot prevents a stale
                notebook from overwriting a deployment that changed meanwhile.
                """
            ),
            _code(
                connection_settings
                + """
MODEL_NAME = "__MODEL_NAME__"  # Set to None to select by label only.
MODEL_LABEL = "__MODEL_LABEL__"
DEPLOYMENT_SLOT = "__DEPLOYMENT_SLOT__"
PACKAGE_VERSION = None  # None selects the latest published package.
DEPLOYMENT_REASON = ""
"""
            ),
            _code(
                _project_setup_source(
                    imports="""
                    from pricing_pipeline.notebook import (  # noqa: E402
                        connect,
                        deploy_package,
                        list_candidate_versions,
                        load_registered_model,
                        open_candidate,
                    )
                    """,
                )
            ),
            _markdown("## Connect and list deployable SQL packages"),
            _code(connect_source),
            _code(
                """
                model = load_registered_model(
                    pricing,
                    model_name=MODEL_NAME,
                    model_label=MODEL_LABEL,
                    deployment_slot=DEPLOYMENT_SLOT,
                    source_root=MODEL_DIR,
                )
                versions = list_candidate_versions(pricing, model=model, technical=True)
                deployable = versions.loc[
                    versions["package_status"].astype(str).str.upper().eq("PUBLISHED")
                ].copy()
                display(
                    deployable.loc[
                        :,
                        [
                            "package_version",
                            "model_version",
                            "model_kind",
                            "model_equivalence_sha256",
                            "data_as_of_date",
                            "manifest_id",
                            "parent_rate_package_id",
                            "current_rate_package_id",
                        ],
                    ]
                )
                """
            ),
            _markdown("## Select and review one published package"),
            _code(
                """
                if deployable.empty:
                    raise LookupError("No published candidate packages were found.")
                selected_package_version = (
                    int(deployable.iloc[0]["package_version"])
                    if PACKAGE_VERSION is None
                    else int(PACKAGE_VERSION)
                )
                if selected_package_version not in set(
                    deployable["package_version"].astype(int)
                ):
                    raise ValueError("PACKAGE_VERSION is not in the displayed published list.")
                reviewed = open_candidate(
                    pricing,
                    model=model,
                    package_version=selected_package_version,
                )
                display(reviewed.technical)
                """
            ),
            _markdown("## Deploy the reviewed package"),
            _code(
                """
                if not DEPLOYMENT_REASON.strip():
                    raise ValueError("Describe the approval for changing the live package.")
                deployment = deploy_package(
                    pricing,
                    package=reviewed,
                    reason=DEPLOYMENT_REASON,
                )
                display(deployment)
                """
            ),
        ]
    )

    scratch = _notebook_document(
        [
            _markdown(
                """
                # __MODEL_LABEL_MARKDOWN__: scratch work

                Use this notebook for source exploration and disposable feature ideas.
                The first section is a complete but disposable data-to-model sandbox:
                load or assemble data, transform it, fit ordinary SuperGLM objects, and
                inspect predictions. It deliberately sorts last and never updates the
                governed model-frame artifact. Move accepted data work into
                `01_data_ingestion.ipynb` and accepted model choices into
                `02_model_training.ipynb`.

                It also provides the temporary grouping workflow: open a published RAW
                candidate in SuperGLM's editor, collapse categorical levels, then export
                the actual per-feature `LevelGrouping` objects for notebook 02.
                """
            ),
            _code(
                connection_settings
                + """
MODEL_NAME = "__MODEL_NAME__"
MODEL_LABEL = "__MODEL_LABEL__"
DEPLOYMENT_SLOT = "__DEPLOYMENT_SLOT__"
SCRATCH_SAMPLE_ROWS = 5_000  # Set to None to use every row.
SCRATCH_RANDOM_SEED = 42
GROUPING_SOURCE_PACKAGE_VERSION = None  # None selects the latest published RAW package.
REPLACE_GROUPING_ARTIFACT = False
"""
            ),
            _code(
                _project_setup_source(
                    imports="""
                    import numpy as np  # noqa: E402
                    import pandas as pd  # noqa: E402
                    from superglm import Categorical, Numeric, Spline, SuperGLM  # noqa: E402
                    from superglm.editor import EditorSession  # noqa: E402

                    from pricing_pipeline.notebook import (  # noqa: E402
                        connect,
                        export_level_groupings,
                        list_candidate_versions,
                        load_registered_model,
                        open_candidate,
                    )
                    """,
                )
            ),
            _code(connect_source),
            _markdown(
                """
                ## Sandbox 1: load or assemble disposable data

                Replace this demo with any SQL query, file read, join, filter, or sample
                you want to investigate. Nothing in this section is saved as the
                governed model-frame handoff.
                """
            ),
            _code(
                """
                rng = np.random.default_rng(SCRATCH_RANDOM_SEED)
                scratch_raw = pd.DataFrame({
                    "__PRIMARY_KEY__": np.arange(1, 501),
                    "__FEATURE_NAME__": rng.normal(size=500),
                    "segment": rng.choice(["A", "B", "C"], size=500),
                })
                scratch_raw["__TARGET_NAME__"] = rng.poisson(
                    np.exp(
                        -0.5
                        + 0.25 * scratch_raw["__FEATURE_NAME__"]
                        + scratch_raw["segment"].map({"A": 0.0, "B": 0.2, "C": -0.1})
                    )
                )
                if SCRATCH_SAMPLE_ROWS is not None and len(scratch_raw) > SCRATCH_SAMPLE_ROWS:
                    scratch_raw = scratch_raw.sample(
                        n=SCRATCH_SAMPLE_ROWS,
                        random_state=SCRATCH_RANDOM_SEED,
                    )
                scratch_raw = scratch_raw.sort_values("__PRIMARY_KEY__").reset_index(drop=True)
                display({"Rows": len(scratch_raw), "Columns": len(scratch_raw.columns)})
                display(scratch_raw.head())
                """
            ),
            _code(
                """
                # Blank ingestion area: replace or extend scratch_raw however you like.
                # scratch_raw = pd.read_csv("...")
                # scratch_raw = pd.read_sql_query("SELECT ...", pricing.engine)
                """
            ),
            _markdown(
                """
                ## Sandbox 2: clean and engineer disposable features

                Work on `scratch_frame` so the originally loaded sample remains easy to
                recover. Copy only accepted transforms into notebook 01.
                """
            ),
            _code(
                """
                scratch_frame = scratch_raw.copy()
                scratch_frame["candidate_transform"] = np.square(
                    scratch_frame["__FEATURE_NAME__"]
                )
                display(scratch_frame.groupby("segment")["candidate_transform"].describe())
                display(scratch_frame.head())
                """
            ),
            _code(
                """
                # Blank feature area: add plots, joins, filters, or alternative columns.
                # scratch_frame["another_candidate"] = ...
                """
            ),
            _markdown(
                """
                ## Sandbox 3: define and fit a disposable model

                Use normal SuperGLM feature objects here. This fits only in memory: it
                does not register a model, create a manifest, build a candidate, or
                publish anything. Copy accepted choices into notebook 02.
                """
            ),
            _code(
                """
                SCRATCH_TARGET = "__TARGET_NAME__"
                SCRATCH_FEATURES = {
                    "__FEATURE_NAME__": Numeric(),
                    "segment": Categorical(),
                    "candidate_transform": Numeric(),
                }
                scratch_X = scratch_frame.loc[:, list(SCRATCH_FEATURES)]
                scratch_y = scratch_frame[SCRATCH_TARGET].astype(float)

                scratch_model = SuperGLM(
                    family="poisson",
                    selection_penalty=0.0,
                    features=SCRATCH_FEATURES,
                ).fit(scratch_X, scratch_y)
                """
            ),
            _markdown("## Sandbox 4: inspect, compare, and iterate"),
            _code(
                """
                scratch_predictions = scratch_model.predict(scratch_X)
                scratch_results = pd.DataFrame({
                    "actual": scratch_y,
                    "prediction": scratch_predictions,
                })
                display({
                    "Rows fitted": len(scratch_results),
                    "Actual mean": float(scratch_results["actual"].mean()),
                    "Predicted mean": float(scratch_results["prediction"].mean()),
                    "Mean absolute error": float(
                        np.mean(
                            np.abs(
                                scratch_results["actual"]
                                - scratch_results["prediction"]
                            )
                        )
                    ),
                })
                display(scratch_results.head())
                """
            ),
            _code(
                """
                # Blank modelling area: try another feature map, family, penalty, or split.
                # alternative_features = {...}
                # alternative_model = SuperGLM(...).fit(...)
                """
            ),
            _markdown(
                """
                ## Optional: create the routine level-grouping artifact

                This section is read-only in SQL but requires `DATABASE_MODE = "remote"`
                because editable candidate bundles are governed by the remote workbench.
                Select levels in the widget and use **Collapse and refit** as often as
                needed, across as many categorical features as needed.
                """
            ),
            _code(
                """
                model = load_registered_model(
                    pricing,
                    model_name=MODEL_NAME,
                    model_label=MODEL_LABEL,
                    deployment_slot=DEPLOYMENT_SLOT,
                    source_root=MODEL_DIR,
                )
                versions = list_candidate_versions(pricing, model=model)
                raw_versions = versions.loc[versions["Kind"].eq("RAW")].copy()
                display(raw_versions)
                if raw_versions.empty:
                    raise LookupError("No published RAW candidate is available for grouping.")
                selected_package_version = (
                    int(raw_versions.iloc[0]["Package"])
                    if GROUPING_SOURCE_PACKAGE_VERSION is None
                    else int(GROUPING_SOURCE_PACKAGE_VERSION)
                )
                if selected_package_version not in set(
                    raw_versions["Package"].astype(int)
                ):
                    raise ValueError(
                        "GROUPING_SOURCE_PACKAGE_VERSION is not in the displayed RAW list."
                    )
                grouping_candidate = open_candidate(
                    pricing,
                    model=model,
                    package_version=selected_package_version,
                )
                """
            ),
            _code(
                """
                grouping_session = EditorSession.from_model(
                    grouping_candidate.bundle.fitted_model,
                    train_data=(
                        grouping_candidate.bundle.X,
                        grouping_candidate.bundle.y,
                        grouping_candidate.bundle.sample_weight,
                        grouping_candidate.bundle.offset,
                    ),
                    cv_report=grouping_candidate.bundle.cv_report,
                )
                display(grouping_session.widget())
                """
            ),
            _markdown(
                """
                ## Export all current groupings

                Run this only after every intended collapse/refit has completed. The
                binary contains actual `LevelGrouping` objects; its JSON sidecar is
                generated integrity and lineage evidence, not an analyst config file.
                """
            ),
            _code(
                """
                grouping_artifact = export_level_groupings(
                    grouping_candidate,
                    editor_session=grouping_session,
                    path=GROUPING_ARTIFACT_PATH,
                    replace=REPLACE_GROUPING_ARTIFACT,
                )
                display(grouping_artifact)
                """
            ),
        ]
    )

    return {
        "01_data_ingestion.ipynb": ingestion,
        "02_model_training.ipynb": training,
        "03_model_editor.ipynb": editor,
        "04_model_deployment.ipynb": deployment,
        "99_scratch_work.ipynb": scratch,
    }


def _notebooks(
    *,
    package_name: str,
    model_name: str,
    model_label: str,
    target_name: str,
    model_type: str,
    deployment_slot: str,
    database_mode: str,
    runtime_module: str | None,
    expected_remote_database: str,
) -> dict[str, str]:
    feature = "feature_1" if target_name != "feature_1" else "feature_2"
    primary_key = "row_id" if target_name != "row_id" else "record_id"
    python_values = {
        "__PACKAGE_NAME__": package_name,
        "__MODEL_NAME__": model_name,
        "__MODEL_LABEL__": model_label,
        "__TARGET_NAME__": target_name,
        "__MODEL_TYPE__": model_type,
        "__DEPLOYMENT_SLOT__": deployment_slot,
        "__FEATURE_NAME__": feature,
        "__PRIMARY_KEY__": primary_key,
        "__DATASET_NAME__": f"{package_name}_model_frame",
    }
    replacements = {token: json.dumps(value)[1:-1] for token, value in python_values.items()}
    replacements["__MODEL_LABEL_MARKDOWN__"] = model_label
    rendered = {}
    for filename, template in _template_notebooks(
        database_mode=database_mode,
        runtime_module=runtime_module,
        expected_remote_database=expected_remote_database,
    ).items():
        rendered[filename] = (
            json.dumps(
                _render_template(template, replacements),
                indent=1,
                ensure_ascii=False,
            )
            + "\n"
        )
    return rendered


def scaffold_pricing_model(options: ScaffoldOptions) -> ScaffoldResult:
    model_name = _model_name(options.model_name)
    package_name = _package_name(
        options.package_name or re.sub(r"_+", "_", model_name.lower()).strip("_")
    )
    target_name = _required(options.target_name, "target_name")
    model_label = _required(
        options.model_label or model_name.replace("_", " ").title(), "model_label"
    )
    model_type = _required(options.model_type, "model_type")
    deployment_slot = _required(
        options.deployment_slot or f"{model_name}_UAT",
        "deployment_slot",
    )
    database_mode = _database_mode(options.database_mode)
    runtime_module = _runtime_module(options.runtime_module)
    expected_remote_database = _expected_remote_database(options.expected_remote_database)
    if database_mode == "remote" and not expected_remote_database:
        raise ValueError("expected_remote_database is required when database_mode='remote'")

    package_dir = options.root / "pricing_models" / package_name
    notebooks = _notebooks(
        package_name=package_name,
        model_name=model_name,
        model_label=model_label,
        target_name=target_name,
        model_type=model_type,
        deployment_slot=deployment_slot,
        database_mode=database_mode,
        runtime_module=runtime_module,
        expected_remote_database=expected_remote_database,
    )
    content = {
        package_dir / "__init__.py": f'"""Pricing notebook package for {model_name}."""\n',
        **{package_dir / filename: source for filename, source in notebooks.items()},
    }
    created = []
    for path, source in content.items():
        if path.exists() and not options.force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        created.append(path)
    return ScaffoldResult(package_name=package_name, created_files=tuple(created))


def parse_args(argv: list[str] | None = None) -> ScaffoldOptions:
    parser = argparse.ArgumentParser(description="Create a pricing-model notebook workflow.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--model-type", default="superglm_poisson")
    parser.add_argument("--deployment-slot")
    parser.add_argument("--package-name")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "TOML defaults file; when omitted, <root>/pricing_scaffold.toml is loaded if present"
        ),
    )
    parser.add_argument(
        "--database-mode",
        choices=("local", "remote"),
        help="override notebook_defaults.database_mode",
    )
    parser.add_argument(
        "--runtime-module",
        help="override notebook_defaults.runtime_module",
    )
    parser.add_argument(
        "--expected-remote-database",
        help="override notebook_defaults.expected_remote_database",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    auto_config = args.root / _DEFAULT_CONFIG_NAME
    config_path = args.config if args.config is not None else auto_config
    if args.config is not None or config_path.is_file():
        try:
            config = load_scaffold_config(config_path)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
    else:
        config = ScaffoldConfig()
    try:
        database_mode = _database_mode(
            args.database_mode if args.database_mode is not None else config.database_mode
        )
        runtime_module = _runtime_module(
            args.runtime_module if args.runtime_module is not None else config.runtime_module
        )
        expected_remote_database = _expected_remote_database(
            args.expected_remote_database
            if args.expected_remote_database is not None
            else config.expected_remote_database
        )
        if database_mode == "remote" and not expected_remote_database:
            raise ValueError("expected_remote_database is required when database_mode='remote'")
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return ScaffoldOptions(
        model_name=args.model_name,
        target_name=args.target_name,
        model_label=args.model_label,
        model_type=args.model_type,
        deployment_slot=args.deployment_slot,
        package_name=args.package_name,
        database_mode=database_mode,
        runtime_module=runtime_module,
        expected_remote_database=expected_remote_database,
        root=args.root,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> None:
    result = scaffold_pricing_model(parse_args(argv))
    for path in result.created_files:
        print(path.as_posix())


if __name__ == "__main__":
    main()
