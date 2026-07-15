from __future__ import annotations

import argparse
import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ScaffoldOptions:
    model_name: str
    target_name: str
    model_label: str | None = None
    model_type: str = "superglm_poisson"
    deployment_slot: str | None = None
    package_name: str | None = None
    root: Path = Path(".")
    force: bool = False


@dataclass(frozen=True)
class ScaffoldResult:
    package_name: str
    created_files: tuple[Path, ...]


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


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [dedent(source).strip() + "\n"],
    }


def _markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [dedent(source).strip() + "\n"],
    }


def _notebook(
    *,
    package_name: str,
    model_name: str,
    model_label: str,
    target_name: str,
    model_type: str,
    deployment_slot: str,
) -> str:
    feature = "feature_1" if target_name != "feature_1" else "feature_2"
    primary_key = "row_id" if target_name != "row_id" else "record_id"
    exposure = "exposure" if target_name != "exposure" else "earned_exposure"
    q = json.dumps
    cells = [
        _markdown(
            f"""
            # {model_label}

            The visible cells own the data, features, transforms, validation choice,
            model, review, and deployment decision. The imported helpers record model
            identity, versions, dataset and split evidence, artifact hashes, lineage,
            and rating-package rows.
            """
        ),
        _code(
            """
            DATABASE_MODE = "local"  # "local" or "remote"
            RUNTIME_MODULE = None  # e.g. "work_runtime.database"; never put secrets here
            EXPECTED_REMOTE_DATABASE = ""
            ALLOW_REMOTE_WRITES = False

            DATA_AS_OF = None  # Or use MODEL.data_as_of_column below.
            RUN_EDITOR = False
            EDIT_REASON = ""
            DEPLOY = False
            DEPLOYMENT_REASON = ""
            """
        ),
        _code(
            f"""
            from datetime import date
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

            import numpy as np  # noqa: E402
            import pandas as pd  # noqa: E402
            from superglm import Numeric, SuperGLM  # noqa: E402

            from pricing_pipeline.models.config import ValidationSplitConfig  # noqa: E402
            from pricing_pipeline.notebook import (  # noqa: E402
                PricingModelSpec,
                build_candidate,
                connect,
                deploy_package,
                open_candidate,
                publish_candidate,
                publish_edits,
                register_model,
            )

            MODEL_DIR = PROJECT_ROOT / "pricing_models/{package_name}"
            """
        ),
        _markdown(
            """
            ## Connect and verify the destination

            Local mode creates persistent SQLite files under `.local`. Remote mode
            obtains its private connection from the work runtime configured outside
            this repository and refuses writes until the expected database matches.
            """
        ),
        _code(
            """
            pricing = connect(
                mode=DATABASE_MODE,
                runtime_module=RUNTIME_MODULE,
                local_root=MODEL_DIR / ".local",
                expected_remote_database=EXPECTED_REMOTE_DATABASE,
                allow_remote_writes=ALLOW_REMOTE_WRITES,
            )
            display(pricing.destination)
            """
        ),
        _markdown(
            """
            ## Load and transform the model frame

            Replace the demo with the normal work query. Keep feature transforms as
            ordinary visible Python and retain the primary key, target, exposure or
            weights, optional split column, and data-as-of column in the final frame.
            """
        ),
        _code(
            f"""
            rng = np.random.default_rng(42)
            frame = pd.DataFrame({{
                {q(primary_key)}: np.arange(1, 101),
                {q(feature)}: rng.normal(size=100),
                {q(exposure)}: rng.uniform(0.25, 1.0, size=100),
                "data_as_of": [date.today()] * 100,
            }})
            frame[{q(target_name)}] = rng.poisson(
                frame[{q(exposure)}] * np.exp(-2.5 + 0.25 * frame[{q(feature)}])
            )
            display({{"Rows": len(frame), "Columns": len(frame.columns)}})
            """
        ),
        _markdown("## Define the model and validation decision"),
        _code(
            f"""
            MODEL = PricingModelSpec(
                name={q(model_name)},
                label={q(model_label)},
                target={q(target_name)},
                model_type={q(model_type)},
                deployment_slot={q(deployment_slot)},
                features=({q(feature)},),
                dataset_name={q(package_name + "_model_frame")},
                source_system="replace_with_source_name",
                pk_columns=({q(primary_key)},),
                exposure_column={q(exposure)},
                data_as_of_column="data_as_of",
                validation=ValidationSplitConfig.kfold(
                    n_splits=5,
                    random_state=42,
                    shuffle=True,
                ),
            )

            def make_model():
                return SuperGLM(
                    family="poisson",
                    selection_penalty=0.0,
                    discrete=True,
                    n_bins=64,
                    features={{{q(feature)}: Numeric()}},
                )

            model = register_model(pricing, MODEL, source_root=MODEL_DIR)
            """
        ),
        _markdown("## Fit and inspect the candidate"),
        _code(
            """
            candidate = build_candidate(
                pricing,
                model=model,
                frame=frame,
                model_factory=make_model,
                data_as_of=DATA_AS_OF,
            )
            candidate.metrics
            """
        ),
        _markdown(
            """
            ## Publish the immutable candidate

            Publication records the audit trail and creates a candidate package. It
            does not change the live deployment.
            """
        ),
        _code(
            """
            published = publish_candidate(pricing, candidate)
            display({
                "Model": published.model_name,
                "Package": published.package_version,
                "State": published.package_status,
            })
            """
        ),
        _markdown("## Optional market edit and explicit review (remote mode only)"),
        _code(
            """
            reviewed = None
            if RUN_EDITOR:
                reviewed = open_candidate(
                    pricing,
                    model=model,
                    package_version=published.package_version,
                )
                display(reviewed.editor())
                if not EDIT_REASON.strip():
                    raise ValueError("Describe the market or underwriting edit.")
                edited = publish_edits(
                    pricing,
                    candidate=reviewed,
                    reason=EDIT_REASON,
                )
                reviewed = open_candidate(
                    pricing,
                    model=model,
                    package_version=edited.package_version,
                )
                display({"Edited package": edited.package_version, "State": edited.package_status})
            """
        ),
        _markdown("## Optional deployment of the reviewed package (remote mode only)"),
        _code(
            """
            if DEPLOY:
                if reviewed is None:
                    reviewed = open_candidate(
                        pricing,
                        model=model,
                        package_version=published.package_version,
                    )
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
    return (
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )


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

    package_dir = options.root / "pricing_models" / package_name
    content = {
        package_dir / "__init__.py": f'"""Pricing notebook package for {model_name}."""\n',
        package_dir / "pricing_model.ipynb": _notebook(
            package_name=package_name,
            model_name=model_name,
            model_label=model_label,
            target_name=target_name,
            model_type=model_type,
            deployment_slot=deployment_slot,
        ),
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
    parser = argparse.ArgumentParser(description="Create one pricing-model notebook.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--model-type", default="superglm_poisson")
    parser.add_argument("--deployment-slot")
    parser.add_argument("--package-name")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return ScaffoldOptions(
        model_name=args.model_name,
        target_name=args.target_name,
        model_label=args.model_label,
        model_type=args.model_type,
        deployment_slot=args.deployment_slot,
        package_name=args.package_name,
        root=args.root,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> None:
    result = scaffold_pricing_model(parse_args(argv))
    for path in result.created_files:
        print(path.as_posix())


if __name__ == "__main__":
    main()
