from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json

import pytest

from pricing_pipeline.publishing.naming import clean_identifier
from pricing_pipeline.publishing.superglm_publication_receipt import (
    OffsetExportContract,
    SuperGLMPublicationReceipt,
    canonical_receipt_bytes,
    load_publication_receipt,
    publication_receipt_sha256,
    write_publication_receipt,
)


def _receipt() -> SuperGLMPublicationReceipt:
    return SuperGLMPublicationReceipt(
        schema_name="superglm_publication_receipt",
        schema_version=1,
        metadata_origin="SUPERGLM_FITTED_MODEL",
        superglm_version="0.10.0",
        extractor_version="1",
        package_metadata={"model": {"family": "poisson", "link": "log", "fit_used_offset": True}},
        term_metadata={
            "TermMonths": {
                "feature_kind": "offset",
                "source_term_name": "Term Months",
                "published_term_name": "TermMonths",
            }
        },
        offset_contract=OffsetExportContract(
            handling="EXPORTED_FACTOR",
            source_factor_name="Term Months",
            published_factor_name="TermMonths",
            source_name="TermMonths",
            label="log(TermMonths / 12)",
        ),
    )


def test_clean_identifier_matches_staging_normalization():
    assert clean_identifier("Term Months") == "Term_Months"
    assert clean_identifier("  A/B + C  ") == "A_B_C"
    assert clean_identifier("") == "unknown"


def test_offset_contract_cross_field_validation():
    OffsetExportContract(handling="NONE")
    OffsetExportContract(
        handling="EXPORTED_FACTOR",
        source_factor_name="Term Months",
        published_factor_name="Term_Months",
        source_name="TermMonths",
        label="log(TermMonths / 12)",
    )
    OffsetExportContract(
        handling="ALREADY_APPLIED_SQL_EXPOSURE",
        source_name="Exposure",
        label="log(Exposure)",
    )

    with pytest.raises(ValueError, match="must be null"):
        OffsetExportContract(handling="NONE", source_name="Exposure")
    with pytest.raises(ValueError, match="required"):
        OffsetExportContract(handling="EXPORTED_FACTOR", source_name="TermMonths")
    with pytest.raises(ValueError, match="must be null"):
        OffsetExportContract(
            handling="ALREADY_APPLIED_SQL_EXPOSURE",
            source_factor_name="Exposure",
            source_name="Exposure",
            label="log(Exposure)",
        )


def test_canonical_receipt_hash_uses_exact_canonical_bytes(tmp_path):
    receipt = _receipt()
    expected = json.dumps(
        receipt.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    assert canonical_receipt_bytes(receipt) == expected
    path = tmp_path / "superglm_publication_receipt.json"
    digest = write_publication_receipt(receipt, path)

    assert path.read_bytes() == expected
    assert digest == publication_receipt_sha256(receipt)
    assert load_publication_receipt(path, expected_sha256=digest) == receipt
    assert load_publication_receipt(path, digest) == receipt


def test_canonical_receipt_rejects_non_finite_package_metadata():
    receipt = _receipt().model_copy(update={"package_metadata": {"bad": float("nan")}})

    with pytest.raises(ValueError, match="non-finite"):
        canonical_receipt_bytes(receipt)


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": datetime(2026, 1, 1)},
        {"bad": Decimal("1.25")},
        {"bad": ("tuple",)},
        {1: "not a string key"},
        {"nested": {"bad": float("-inf")}},
    ],
)
def test_receipt_rejects_non_json_native_package_metadata(metadata):
    payload = _receipt().model_dump(mode="python")
    payload["package_metadata"] = metadata

    with pytest.raises(ValueError, match="metadata"):
        SuperGLMPublicationReceipt(**payload)


def test_receipt_rejects_nested_non_finite_term_metadata():
    payload = _receipt().model_dump(mode="python")
    payload["term_metadata"] = {"TermMonths": {"nested": [1.0, float("inf")]}}

    with pytest.raises(ValueError, match="metadata"):
        SuperGLMPublicationReceipt(**payload)


def test_receipt_metadata_cannot_mutate_after_construction():
    receipt = _receipt()
    digest = publication_receipt_sha256(receipt)

    with pytest.raises(TypeError):
        receipt.package_metadata["model"]["family"] = "gamma"

    assert publication_receipt_sha256(receipt) == digest


def test_receipt_loader_rejects_noncanonical_equivalent_json(tmp_path):
    receipt = _receipt()
    canonical_digest = publication_receipt_sha256(receipt)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        load_publication_receipt(path, expected_sha256=canonical_digest)
