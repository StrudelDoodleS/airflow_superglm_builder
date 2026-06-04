from __future__ import annotations

from textwrap import dedent

from pricing_models.mtpl_frequency.spec import MODEL_CONFIG, MODEL_SPEC


def _write_model_toml(
    package_dir,
    *,
    model_key: str,
    model_label: str | None = None,
    target_name: str = "target",
) -> None:
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.toml").write_text(
        dedent(
            f"""\
            model_key = "{model_key}"
            model_label = "{model_label or model_key.title()}"
            target_name = "{target_name}"
            model_type = "superglm_poisson"
            deployment_slot = "{model_key}_UAT"
            default_package_status = "PUBLISHED"

            [validation_split]
            method = "train_test_split"
            test_size = 0.2
            random_state = 42
            shuffle = true
            materialize = false
            """
        ),
        encoding="utf-8",
    )


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


def test_model_configs_are_available_from_registry():
    from pricing_models.registry import get_model_config

    assert get_model_config("MTPL_FREQ") == MODEL_CONFIG


def test_model_config_registry_discovers_toml_without_importing_specs(tmp_path, monkeypatch):
    models_root = tmp_path / "pricing_models"
    package_dir = models_root / "lazy_model"
    _write_model_toml(package_dir, model_key="LAZY_MODEL", model_label="Lazy model")
    (package_dir / "spec.py").write_text(
        "raise RuntimeError('spec import should not happen for config lookup')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(tmp_path)

    from pricing_models.registry import get_model_config, model_keys

    assert model_keys(models_root=models_root) == ("LAZY_MODEL",)
    config = get_model_config("LAZY_MODEL", models_root=models_root)
    assert config.model_key == "LAZY_MODEL"
    assert config.model_label == "Lazy model"


def test_model_spec_registry_lazy_imports_only_selected_package(tmp_path, monkeypatch):
    models_root = tmp_path / "pricing_models"
    selected_dir = models_root / "selected_model"
    poison_dir = models_root / "poison_model"
    _write_model_toml(selected_dir, model_key="SELECTED_MODEL")
    _write_model_toml(poison_dir, model_key="POISON_MODEL")
    (selected_dir / "spec.py").write_text(
        dedent(
            """\
            from pricing_pipeline.models.spec import DatasetSpec, ModelSpec

            def build_model():
                return object()

            def build_training_frame(raw):
                return raw

            MODEL_SPEC = ModelSpec(
                model_key="SELECTED_MODEL",
                model_label="Selected model",
                target_name="target",
                model_type="superglm_poisson",
                experiment_name="pricing-selected",
                deployment_slot="SELECTED_MODEL_UAT",
                dataset=DatasetSpec(
                    dataset_name="selected_training",
                    source_system="sql_server",
                    manifest_sql="SELECT 1",
                    pk_columns=("id",),
                    target_column="target",
                ),
                training_sql="SELECT 1",
                feature_columns=("rating_factor",),
                build_model=build_model,
                build_training_frame=build_training_frame,
                package_status="PUBLISHED",
            )
            """
        ),
        encoding="utf-8",
    )
    (poison_dir / "spec.py").write_text(
        "raise RuntimeError('unselected spec import should not happen')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(tmp_path)

    from pricing_models.registry import get_model_spec

    spec = get_model_spec(
        "SELECTED_MODEL",
        models_root=models_root,
        package_prefix="pricing_models",
    )

    assert spec.model_key == "SELECTED_MODEL"


def test_model_config_registry_rejects_duplicate_model_keys(tmp_path):
    models_root = tmp_path / "pricing_models"
    _write_model_toml(models_root / "first_model", model_key="DUPLICATE_MODEL")
    _write_model_toml(models_root / "second_model", model_key="DUPLICATE_MODEL")

    from pricing_models.registry import model_keys

    try:
        model_keys(models_root=models_root)
    except ValueError as exc:
        assert "Duplicate model_key 'DUPLICATE_MODEL'" in str(exc)
    else:
        raise AssertionError("duplicate model keys should fail registry discovery")


def test_mtpl_frequency_model_config_matches_spec_identity():
    assert MODEL_CONFIG.model_key == MODEL_SPEC.model_key
    assert MODEL_CONFIG.model_label == MODEL_SPEC.model_label
    assert MODEL_CONFIG.target_name == MODEL_SPEC.target_name
    assert MODEL_CONFIG.model_type == MODEL_SPEC.model_type
    assert MODEL_CONFIG.deployment_slot == MODEL_SPEC.deployment_slot
    assert MODEL_CONFIG.default_package_status == "PUBLISHED"
