from __future__ import annotations

import ast
import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pricing_pipeline.models.config import ModelBuildConfig, load_model_build_config
from pricing_pipeline.models.spec import ModelSpec


MODELS_ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_PREFIX = "pricing_models"


@dataclass(frozen=True)
class ModelRegistryEntry:
    package_name: str
    config_path: Path
    config: ModelBuildConfig

    @property
    def spec_path(self) -> Path:
        return self.config_path.with_name("spec.py")


def _models_root(models_root: Path | None) -> Path:
    return (models_root or MODELS_ROOT).resolve()


@lru_cache(maxsize=None)
def _discover_entries_cached(models_root: str) -> tuple[ModelRegistryEntry, ...]:
    root = Path(models_root)
    entries: list[ModelRegistryEntry] = []
    by_key: dict[str, ModelRegistryEntry] = {}
    for config_path in sorted(root.glob("*/model.toml")):
        entry = ModelRegistryEntry(
            package_name=config_path.parent.name,
            config_path=config_path,
            config=load_model_build_config(config_path),
        )
        existing = by_key.get(entry.config.model_name)
        if existing is not None:
            raise ValueError(
                f"Duplicate model_name {entry.config.model_name!r} in "
                f"{existing.config_path.as_posix()} and {entry.config_path.as_posix()}"
            )
        by_key[entry.config.model_name] = entry
        entries.append(entry)
    return tuple(entries)


def _discover_entries(models_root: Path | None = None) -> tuple[ModelRegistryEntry, ...]:
    return _discover_entries_cached(str(_models_root(models_root)))


def _entries_by_key(models_root: Path | None = None) -> dict[str, ModelRegistryEntry]:
    return {entry.config.model_name: entry for entry in _discover_entries(models_root)}


def model_names(*, models_root: Path | None = None) -> tuple[str, ...]:
    return tuple(sorted(_entries_by_key(models_root)))


def _target_defines_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, ast.Tuple | ast.List):
        return any(_target_defines_name(item, name) for item in target.elts)
    return False


def _spec_file_defines_model_spec(spec_path: Path) -> bool:
    try:
        source = spec_path.read_text(encoding="utf-8")
    except OSError:
        return False

    try:
        tree = ast.parse(source, filename=spec_path.as_posix())
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            _target_defines_name(target, "MODEL_SPEC") for target in node.targets
        ):
            return True
        if isinstance(node, ast.AnnAssign) and _target_defines_name(node.target, "MODEL_SPEC"):
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_name = alias.asname or alias.name
                if imported_name == "MODEL_SPEC":
                    return True
    return False


def model_spec_names(*, models_root: Path | None = None) -> tuple[str, ...]:
    """Return model names that can be run through ModelSpec-based tooling."""
    return tuple(
        sorted(
            entry.config.model_name
            for entry in _discover_entries(models_root)
            if _spec_file_defines_model_spec(entry.spec_path)
        )
    )


def _unknown_model_error(model_name: str, models_root: Path | None = None) -> ValueError:
    choices = ", ".join(model_names(models_root=models_root))
    if choices:
        return ValueError(f"Unknown model name {model_name!r}. Choices: {choices}")
    return ValueError(f"Unknown model name {model_name!r}. No model.toml files found")


def get_model_config(
    model_name: str,
    *,
    models_root: Path | None = None,
) -> ModelBuildConfig:
    try:
        return _entries_by_key(models_root)[model_name].config
    except KeyError as exc:
        raise _unknown_model_error(model_name, models_root) from exc


def _load_spec_from_path(entry: ModelRegistryEntry) -> ModelSpec:
    module_name = f"_pricing_model_registry_{entry.package_name}_{abs(hash(entry.spec_path))}"
    spec = importlib.util.spec_from_file_location(module_name, entry.spec_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import model spec from {entry.spec_path.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL_SPEC


def get_model_spec(
    model_name: str,
    *,
    models_root: Path | None = None,
    package_prefix: str = DEFAULT_PACKAGE_PREFIX,
) -> ModelSpec:
    try:
        entry = _entries_by_key(models_root)[model_name]
    except KeyError as exc:
        raise _unknown_model_error(model_name, models_root) from exc

    if _models_root(models_root) == MODELS_ROOT and package_prefix == DEFAULT_PACKAGE_PREFIX:
        module = importlib.import_module(f"{package_prefix}.{entry.package_name}.spec")
        model_spec = module.MODEL_SPEC
    else:
        model_spec = _load_spec_from_path(entry)

    if model_spec.model_name != model_name:
        raise ValueError(
            f"{entry.spec_path.as_posix()} defines MODEL_SPEC.model_name "
            f"{model_spec.model_name!r}, expected {model_name!r}"
        )
    return model_spec
