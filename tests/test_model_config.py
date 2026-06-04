from pathlib import Path

import pytest

from pricing_pipeline.models.config import (
    ModelBuildConfig,
    ValidationSplitConfig,
    load_model_build_config,
)


def test_load_model_build_config_reads_stable_metadata(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'model_label = "Motor frequency"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
                'default_package_status = "PUBLISHED"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_build_config(path)

    assert config == ModelBuildConfig(
        model_key="MTPL_FREQ",
        model_label="Motor frequency",
        target_name="ClaimNb",
        model_type="superglm_poisson",
        deployment_slot="MTPL_FREQ_UAT",
        default_package_status="PUBLISHED",
        validation_split=ValidationSplitConfig.kfold(),
    )


def test_load_model_build_config_reads_train_test_validation_split(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'model_label = "Motor frequency"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
                'default_package_status = "PUBLISHED"',
                "",
                "[validation_split]",
                'method = "train_test_split"',
                "test_size = 0.25",
                "random_state = 123",
                "shuffle = true",
                "stratify_column = \"ClaimNb\"",
                "materialize = true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_build_config(path)

    assert config.validation_split == ValidationSplitConfig.train_test_split(
        test_size=0.25,
        random_state=123,
        shuffle=True,
        stratify_column="ClaimNb",
        materialize=True,
    )


def test_load_model_build_config_rejects_unknown_validation_split_method(
    tmp_path: Path,
):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'model_label = "Motor frequency"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
                'default_package_status = "PUBLISHED"',
                "",
                "[validation_split]",
                'method = "monte_carlo"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation_split.method"):
        load_model_build_config(path)


def test_load_model_build_config_rejects_missing_required_field(tmp_path: Path):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_label"):
        load_model_build_config(path)


def test_load_model_build_config_rejects_non_published_default_status(
    tmp_path: Path,
):
    path = tmp_path / "model.toml"
    path.write_text(
        "\n".join(
            [
                'model_key = "MTPL_FREQ"',
                'model_label = "Motor frequency"',
                'target_name = "ClaimNb"',
                'model_type = "superglm_poisson"',
                'deployment_slot = "MTPL_FREQ_UAT"',
                'default_package_status = "DRAFT"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_package_status"):
        load_model_build_config(path)
