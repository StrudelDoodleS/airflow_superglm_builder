from __future__ import annotations

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
        existing = by_key.get(entry.config.model_key)
        if existing is not None:
            raise ValueError(
                f"Duplicate model_key {entry.config.model_key!r} in "
                f"{existing.config_path.as_posix()} and {entry.config_path.as_posix()}"
            )
        by_key[entry.config.model_key] = entry
        entries.append(entry)
    return tuple(entries)


def _discover_entries(models_root: Path | None = None) -> tuple[ModelRegistryEntry, ...]:
    return _discover_entries_cached(str(_models_root(models_root)))


def _entries_by_key(models_root: Path | None = None) -> dict[str, ModelRegistryEntry]:
    return {entry.config.model_key: entry for entry in _discover_entries(models_root)}


def model_keys(*, models_root: Path | None = None) -> tuple[str, ...]:
    return tuple(sorted(_entries_by_key(models_root)))


def _unknown_model_error(model_key: str, models_root: Path | None = None) -> ValueError:
    choices = ", ".join(model_keys(models_root=models_root))
    if choices:
        return ValueError(f"Unknown model key {model_key!r}. Choices: {choices}")
    return ValueError(f"Unknown model key {model_key!r}. No model.toml files found")


def get_model_config(
    model_key: str,
    *,
    models_root: Path | None = None,
) -> ModelBuildConfig:
    try:
        return _entries_by_key(models_root)[model_key].config
    except KeyError as exc:
        raise _unknown_model_error(model_key, models_root) from exc


def _load_spec_from_path(entry: ModelRegistryEntry) -> ModelSpec:
    module_name = f"_pricing_model_registry_{entry.package_name}_{abs(hash(entry.spec_path))}"
    spec = importlib.util.spec_from_file_location(module_name, entry.spec_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import model spec from {entry.spec_path.as_posix()}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODEL_SPEC


def get_model_spec(
    model_key: str,
    *,
    models_root: Path | None = None,
    package_prefix: str = DEFAULT_PACKAGE_PREFIX,
) -> ModelSpec:
    try:
        entry = _entries_by_key(models_root)[model_key]
    except KeyError as exc:
        raise _unknown_model_error(model_key, models_root) from exc

    if _models_root(models_root) == MODELS_ROOT and package_prefix == DEFAULT_PACKAGE_PREFIX:
        module = importlib.import_module(f"{package_prefix}.{entry.package_name}.spec")
        model_spec = module.MODEL_SPEC
    else:
        model_spec = _load_spec_from_path(entry)

    if model_spec.model_key != model_key:
        raise ValueError(
            f"{entry.spec_path.as_posix()} defines MODEL_SPEC.model_key "
            f"{model_spec.model_key!r}, expected {model_key!r}"
        )
    return model_spec
