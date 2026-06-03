from __future__ import annotations

import math

from scripts import seed_demo_model_variants


def test_demo_packages_cover_same_model_more_data_new_model_and_manual_vehage_uplift():
    packages = seed_demo_model_variants.demo_packages()

    model_names = {package.model_name for package in packages}
    assert {"MTPL_FREQ_DEMO", "MTPL_SEV_DEMO"}.issubset(model_names)

    frequency_versions = [
        package for package in packages if package.model_name == "MTPL_FREQ_DEMO"
    ]
    assert [package.model_version for package in frequency_versions] == [
        "20260429_v1_base",
        "20260429_v2_more_data",
        "20260429_v3_manual_vehage_uplift",
    ]
    assert {package.model_type for package in frequency_versions} == {
        "superglm_poisson_demo"
    }

    v1_row_count = sum(
        level.record_count for term in frequency_versions[0].terms for level in term.levels
    )
    v2_row_count = sum(
        level.record_count for term in frequency_versions[1].terms for level in term.levels
    )
    assert frequency_versions[0].data_slice_name == "frequency_policy_sample"
    assert frequency_versions[1].data_slice_name == "frequency_expanded_policy_sample"
    assert v2_row_count > v1_row_count
    vehage_term = next(
        term for term in frequency_versions[0].terms if term.term_name == "VehAge"
    )
    assert vehage_term.term_type == "DISCRETIZED_SPLINE_1D"
    assert vehage_term.level_set_type == "SPLINE_GRID_1D"

    v2_vehage = seed_demo_model_variants.package_level_multiplier(
        frequency_versions[1],
        term_name="VehAge",
        level_code="[10, 20)",
    )
    v3_vehage = seed_demo_model_variants.package_level_multiplier(
        frequency_versions[2],
        term_name="VehAge",
        level_code="[10, 20)",
    )
    assert math.isclose(v3_vehage, v2_vehage * 1.10)


def test_build_staging_frames_for_demo_package_uses_expected_schema():
    package = seed_demo_model_variants.demo_packages()[0]

    export_df, rate_df, level_df = seed_demo_model_variants.build_staging_frames(
        package,
        created_by="pytest",
    )

    assert export_df.to_dict("records") == [
        {
            "export_id": package.export_id,
            "model_name": package.model_name,
            "model_version": package.model_version,
            "base_rate": package.base_rate,
            "effective_from_date": package.effective_from,
            "effective_to_date": None,
            "source_file": f"demo://{package.data_slice_name}",
            "created_by": "pytest",
        }
    ]
    assert set(rate_df.columns) == {
        "export_id",
        "row_id",
        "term_name",
        "term_type",
        "sequence_no",
        "cell_key_text",
        "multiplier",
        "log_coefficient",
        "exposure_weight",
        "record_count",
        "is_reference",
        "is_default",
    }
    assert set(level_df.columns) == {
        "export_id",
        "row_id",
        "position_no",
        "feature_name",
        "feature_value_type",
        "level_set_name",
        "level_set_type",
        "level_code",
        "level_label",
        "order_index",
        "lower_bound",
        "upper_bound",
        "representative_value",
        "is_missing",
        "is_other",
    }
    assert len(rate_df) == len(level_df)
    assert rate_df["row_id"].tolist() == list(range(1, len(rate_df) + 1))
    assert level_df["position_no"].eq(1).all()
    assert math.isclose(
        rate_df.loc[0, "log_coefficient"],
        math.log(rate_df.loc[0, "multiplier"]),
    )


def test_demo_manifest_id_is_stable_and_data_specific():
    package = seed_demo_model_variants.demo_packages()[0]

    assert seed_demo_model_variants.manifest_id_for_package(package) == (
        "demo__frequency_policy_sample__20260429"
    )


def test_seed_package_registers_model_before_staging(monkeypatch):
    package = seed_demo_model_variants.demo_packages()[0]
    calls = []

    def fake_ensure_pricing_model(engine, **kwargs):
        calls.append(("ensure_model", kwargs))
        return 17

    def fake_insert_staging_frames(engine, args, export_df, rate_df, level_df):
        calls.append(("stage", getattr(args, "model_id", None)))
        assert getattr(args, "model_id", None) == 17

    monkeypatch.setattr(
        seed_demo_model_variants,
        "ensure_pricing_model",
        fake_ensure_pricing_model,
    )
    monkeypatch.setattr(
        seed_demo_model_variants,
        "insert_staging_frames",
        fake_insert_staging_frames,
    )
    monkeypatch.setattr(
        seed_demo_model_variants,
        "publish_rating_package",
        lambda *args, **kwargs: calls.append(("publish", kwargs)) or 42,
    )
    monkeypatch.setattr(
        seed_demo_model_variants,
        "ensure_demo_manifest",
        lambda *args, **kwargs: "manifest-1",
    )
    monkeypatch.setattr(
        seed_demo_model_variants,
        "deploy_rate_package",
        lambda *args, **kwargs: calls.append(("deploy", kwargs)),
    )
    monkeypatch.setattr(
        seed_demo_model_variants,
        "record_model_run",
        lambda *args, **kwargs: calls.append(("record_run", kwargs)),
    )

    rate_package_id = seed_demo_model_variants.seed_package(
        object(),
        package,
        created_by="pytest",
    )

    assert rate_package_id == 42
    assert [call[0] for call in calls] == [
        "ensure_model",
        "stage",
        "publish",
        "deploy",
        "record_run",
    ]
