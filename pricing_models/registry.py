from __future__ import annotations

from pricing_pipeline.models.spec import ModelSpec
from pricing_models.mtpl_frequency.spec import MODEL_SPEC as MTPL_FREQUENCY_SPEC


MODEL_SPECS: dict[str, ModelSpec] = {
    MTPL_FREQUENCY_SPEC.model_key: MTPL_FREQUENCY_SPEC,
}


def model_keys() -> tuple[str, ...]:
    return tuple(sorted(MODEL_SPECS))


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        choices = ", ".join(model_keys())
        raise ValueError(f"Unknown model key {model_key!r}. Choices: {choices}") from exc
