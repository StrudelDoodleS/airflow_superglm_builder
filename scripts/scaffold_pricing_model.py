from __future__ import annotations

import argparse
import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path


_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TEMPLATE_TOKEN = re.compile(r"__[A-Z][A-Z0-9_]*__")
_NOTEBOOK_TEMPLATE = Path(__file__).parent / "templates" / "pricing_model.ipynb"


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


def _render_template(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, str):
        return _TEMPLATE_TOKEN.sub(lambda match: replacements[match.group()], value)
    if isinstance(value, list):
        return [_render_template(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _render_template(item, replacements) for key, item in value.items()}
    return value


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
    template = json.loads(_NOTEBOOK_TEMPLATE.read_text(encoding="utf-8"))
    return json.dumps(_render_template(template, replacements), indent=1, ensure_ascii=False) + "\n"


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
