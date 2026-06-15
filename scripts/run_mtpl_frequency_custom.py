from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_models.mtpl_frequency.data import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    prepare_source_data,
)
from pricing_models.mtpl_frequency.modeling import train_validate_export_model  # noqa: E402
from pricing_models.mtpl_frequency.spec import MODEL_CONFIG  # noqa: E402
from pricing_pipeline.orchestration.airflow_run_metadata import (  # noqa: E402
    merge_prepared_payload_metadata,
)
from pricing_pipeline.orchestration.completed_build_helpers import (  # noqa: E402
    effective_from_for_run,
)
from pricing_pipeline.orchestration.publish_completed_build import (  # noqa: E402
    publish_completed_model_build,
)
from pricing_pipeline.orchestration.run_context import run_key_for_value  # noqa: E402
from pricing_pipeline.publishing.model_registry import ensure_pricing_model  # noqa: E402
from scripts.pricing_db import get_runtime, load_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the explicit freMTPL custom model path without Airflow: "
            "prepare source data, train/export, create a frame manifest, and publish."
        )
    )
    parser.add_argument(
        "--runtime-module",
        default=None,
        help=(
            "Importable Python module that provides get_engine(database=None), "
            "get_schema_names(), and optional get_runtime_settings()."
        ),
    )
    parser.add_argument(
        "--effective-from",
        default=None,
        help="Model effective date. Defaults to today for direct manual runs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory for run artifacts. Defaults to PRICING_MTPL_OUTPUT_DIR "
            "or state/mtpl_frequency."
        ),
    )
    parser.add_argument("--created-by", default="mtpl_frequency_custom")
    return parser.parse_args()


def _output_root(value: str | None) -> Path:
    if value:
        return Path(value)
    return Path(os.environ.get("PRICING_MTPL_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT))


def run_mtpl_frequency_custom(
    *,
    runtime_module: str | None = None,
    effective_from: str | None = None,
    output_root: str | Path | None = None,
    created_by: str = "mtpl_frequency_custom",
) -> dict[str, Any]:
    load_env()
    runtime = get_runtime(runtime_module)
    engine = runtime.get_engine()

    effective = effective_from_for_run(effective_from)
    run_key = run_key_for_value(f"python__{effective}")
    root = Path(output_root) if output_root is not None else _output_root(None)
    metadata = {
        "run_key": run_key,
        "output_dir": str(root / run_key),
        "effective_from": effective,
        "data_as_of_date": effective,
    }

    ensure_pricing_model(
        engine,
        model_key=MODEL_CONFIG.model_key,
        model_label=MODEL_CONFIG.model_label,
        target_name=MODEL_CONFIG.target_name,
        model_type=MODEL_CONFIG.model_type,
        created_by=created_by,
    )
    prepared_payload = prepare_source_data(
        engine,
        run_key=metadata["run_key"],
        output_dir=metadata["output_dir"],
    )
    prepared = merge_prepared_payload_metadata(metadata, prepared_payload)
    completed_build = train_validate_export_model(
        prepared,
        engine=engine,
        settings=runtime.settings,
        created_by=created_by,
    )
    published = publish_completed_model_build(
        engine,
        settings=runtime.settings,
        model_config=MODEL_CONFIG,
        dataset=None,
        completed_build=completed_build,
        created_by=created_by,
    )
    return published.to_dict()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    result = run_mtpl_frequency_custom(
        runtime_module=args.runtime_module,
        effective_from=args.effective_from,
        output_root=args.output_root,
        created_by=args.created_by,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
