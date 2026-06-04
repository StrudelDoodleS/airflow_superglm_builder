from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_models.demo_custom_publish.data import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_OUTPUT_DIR,
    PK_COLUMNS,
    SOURCE_SYSTEM,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    build_demo_training_frame,
    materialize_training_source,
    training_table_for_run,
    write_training_frame,
)
from pricing_models.demo_custom_publish.modeling import (  # noqa: E402
    build_final_model_frame,
    effective_from_for_run,
    export_superglm_completed_build,
    trained_model_version_for_export,
)
from pricing_models.demo_custom_publish.spec import MODEL_CONFIG  # noqa: E402
from pricing_pipeline.data.manifest import (  # noqa: E402
    ModelFrameManifestSpec,
    create_model_frame_manifest_with_split,
)
from pricing_pipeline.orchestration.publish_completed_build import (  # noqa: E402
    publish_completed_model_build,
)
from pricing_pipeline.publishing.rating_export import build_export_id  # noqa: E402
from pricing_pipeline.publishing.model_registry import ensure_pricing_model  # noqa: E402
from scripts.pricing_db import get_runtime  # noqa: E402


def default_output_dir() -> Path:
    return Path(os.environ.get("PRICING_DEMO_CUSTOM_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


def run_demo_custom_publish(
    *,
    output_dir: str | Path | None = None,
    created_by: str = "demo_custom_publish",
    runtime_module: str | None = None,
) -> dict[str, object]:
    runtime = get_runtime(runtime_module)
    engine = runtime.get_engine()
    resolved_output_dir = default_output_dir() if output_dir is None else Path(output_dir)

    ensure_pricing_model(
        engine,
        model_key=MODEL_CONFIG.model_key,
        model_label=MODEL_CONFIG.model_label,
        target_name=MODEL_CONFIG.target_name,
        model_type=MODEL_CONFIG.model_type,
        created_by=created_by,
    )

    frame = build_demo_training_frame()
    final_frame = build_final_model_frame(frame)
    effective_from = effective_from_for_run()
    export_id = build_export_id(
        MODEL_CONFIG.model_key,
        f"python__{effective_from}",
    )
    model_version = trained_model_version_for_export(
        engine,
        model_key=MODEL_CONFIG.model_key,
        export_id=export_id,
    )
    table_name = training_table_for_run(export_id)
    materialize_training_source(engine, frame, table_name=table_name)
    write_training_frame(final_frame, resolved_output_dir / export_id)

    completed_build = export_superglm_completed_build(
        final_frame,
        output_dir=resolved_output_dir,
        model_version=model_version,
        effective_from=effective_from,
        created_by=created_by,
        export_id=export_id,
    )
    manifest = create_model_frame_manifest_with_split(
        engine,
        frame=final_frame,
        spec=ModelFrameManifestSpec(
            dataset_name=DATASET_NAME,
            source_system=SOURCE_SYSTEM,
            data_as_of_date=effective_from,
            pk_columns=PK_COLUMNS,
            target_column=TARGET_COLUMN,
            weight_column=WEIGHT_COLUMN,
        ),
        validation_split=MODEL_CONFIG.validation_split,
        validation_split_artifact_root=runtime.settings.validation_split_artifact_root,
        created_by=created_by,
    )
    completed_build["manifest_id"] = manifest.manifest_id
    completed_build["split_set_id"] = manifest.split_set_id

    return publish_completed_model_build(
        engine,
        settings=runtime.settings,
        model_config=MODEL_CONFIG,
        dataset=None,
        completed_build=completed_build,
        created_by=created_by,
    ).to_dict()


def main() -> None:
    result = run_demo_custom_publish()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
