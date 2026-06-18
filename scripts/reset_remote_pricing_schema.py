"""Drop and reapply owned pricing schema objects in a target SQL Server database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_pipeline.infra.reset_schema import (  # noqa: E402
    CONFIRMATION_FLAG,
    DEFAULT_RESET_SCHEMAS,
    reset_and_reseed_schema,
)
from scripts.pricing_db import get_runtime, load_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-module",
        default=None,
        help=(
            "Importable Python module that provides get_engine(database=None), "
            "get_schema_names(), and optional get_runtime_settings()."
        ),
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("db/migrations"),
        help="Directory containing versioned SQL Server migration files.",
    )
    parser.add_argument(
        "--expected-database",
        required=True,
        help="Database name that must match DB_NAME() before any destructive action.",
    )
    parser.add_argument(
        "--schemas",
        nargs="*",
        default=list(DEFAULT_RESET_SCHEMAS),
        help=(
            "Owned schemas to drop/reseed, in this order: pricing pricing_stg mlops. "
            "Defaults to pricing pricing_stg mlops."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually drop owned objects and reapply migrations. Default is dry-run.",
    )
    parser.add_argument(
        CONFIRMATION_FLAG,
        dest="confirmed_destructive_reset",
        action="store_true",
        help="Required with --execute to confirm the destructive reset.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.execute and not args.confirmed_destructive_reset:
        raise SystemExit(f"Refusing to execute destructive reset without {CONFIRMATION_FLAG}")


def _resolve_schema_dir(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    load_env()

    engine = get_runtime(args.runtime_module).get_engine()
    result = reset_and_reseed_schema(
        engine,
        migrations_dir=_resolve_schema_dir(args.schema_dir),
        expected_database=args.expected_database,
        schema_names=tuple(args.schemas),
        execute=args.execute,
    )

    print(f"dry_run={str(result.dry_run).lower()}")
    print(f"database={result.actual_database}")
    print(f"schemas={','.join(result.schemas)}")
    print(f"drop_batches={result.drop_batch_count}")
    if result.dry_run:
        print("no objects dropped; re-run with --execute and confirmation flag")
    else:
        print(f"applied_migrations={len(result.applied_migrations)}")
        for migration_name in result.applied_migrations:
            print(f"apply {migration_name}")
    print("done")


if __name__ == "__main__":
    main()
