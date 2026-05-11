from __future__ import annotations

from pricing_pipeline.fremtpl import load_fremtpl_raw
from pricing_pipeline.manifest import FREMTPL_DATASET_NAME, create_fremtpl_manifest
from pricing_pipeline.model_spec import DatasetSpec


FREMTPL_DATASET_SPEC = DatasetSpec(
    dataset_name=FREMTPL_DATASET_NAME,
    load_raw=load_fremtpl_raw,
    create_manifest=create_fremtpl_manifest,
)
