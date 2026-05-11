from __future__ import annotations


def test_mtpl_frequency_spec_lives_in_model_package():
    from pricing_models.mtpl_frequency.spec import MODEL_SPEC

    assert MODEL_SPEC.model_key == "MTPL_FREQ"
    assert MODEL_SPEC.target_name == "ClaimNb"
    assert MODEL_SPEC.model_type == "superglm_poisson"
    assert MODEL_SPEC.experiment_name == "pricing-mtpl-frequency"
    assert MODEL_SPEC.deployment_slot == "MTPL_FREQ_UAT"
    assert MODEL_SPEC.dataset.dataset_name == "freMTPL2freq"
    assert MODEL_SPEC.build_model.__module__ == "pricing_models.mtpl_frequency.training"
    assert (
        MODEL_SPEC.build_training_frame.__module__
        == "pricing_models.mtpl_frequency.training"
    )


def test_model_specs_are_available_from_registry():
    from pricing_models.registry import get_model_spec, model_keys

    assert "MTPL_FREQ" in model_keys()
    assert get_model_spec("MTPL_FREQ").model_key == "MTPL_FREQ"
