from pathlib import Path

import pytest

from pricing_pipeline.models.config import ModelBuildConfig, load_model_build_config


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
    )


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
