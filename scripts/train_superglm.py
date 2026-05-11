from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.infra.config import Settings  # noqa: E402
from pricing_pipeline.infra.db import get_engine  # noqa: E402
from pricing_pipeline.infra.mlflow_tracking import configure_mlflow  # noqa: E402
from pricing_models.mtpl_frequency.training import train_superglm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/opt/pricing/state/rating_exports/manual"),
    )
    parser.add_argument(
        "--experiment",
        default="pricing-mtpl-frequency",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env(os.environ)
    configure_mlflow(settings.mlflow_tracking_uri)
    result = train_superglm(
        get_engine(settings),
        model_dir=args.model_dir,
        mlflow_experiment=args.experiment,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
