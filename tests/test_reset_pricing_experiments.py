from __future__ import annotations

from scripts import reset_pricing_experiments


def test_reset_sql_deletes_dependent_pricing_tables_before_parent_tables():
    required_order = [
        "DELETE FROM mlops.MODEL_RUN_SPLIT_SET",
        "DELETE FROM mlops.MODEL_RUN_DATASET",
        "DELETE FROM mlops.MODEL_RUN_METRIC",
        "DELETE FROM pricing.CV_FOLD_METRIC",
        "DELETE FROM pricing.MODEL_RUN",
        "DELETE FROM pricing.PRICING_MODEL_DEPLOYMENT",
        "DELETE FROM pricing.PRICING_PACKAGE_POINTER",
        "DELETE FROM pricing.PRICING_COMPILED_1D_RATE_BAND",
        "DELETE FROM pricing.PRICING_COMPILED_RATE_CELL",
        "DELETE FROM pricing.PRICING_RATE_CELL_LEVEL",
        "DELETE FROM pricing.PRICING_RATE_CELL",
        "DELETE FROM pricing.PRICING_TERM_FEATURE",
        "DELETE FROM pricing.PRICING_TERM",
        "DELETE FROM pricing.PRICING_RATE_PACKAGE",
        "DELETE FROM pricing.PRICING_FEATURE_LEVEL",
        "DELETE FROM pricing.PRICING_FEATURE_LEVEL_SET",
        "DELETE FROM pricing.PRICING_FEATURE",
        "DELETE FROM pricing.CV_FOLD",
        "DELETE FROM pricing.CV_SPLIT_SET",
        "DELETE FROM pricing.DATASET_COLUMN",
        "DELETE FROM pricing.DATASET_MANIFEST",
        "DELETE FROM pricing_stg.STG_CELL_LEVEL",
        "DELETE FROM pricing_stg.STG_RATE_CELL",
        "DELETE FROM pricing_stg.STG_RATING_EXPORT",
        "DELETE FROM pricing.PRICING_MODEL",
    ]

    actual_order = [
        statement.strip()
        for statement in reset_pricing_experiments.RESET_SQL.strip().split(";")
        if statement.strip()
    ]
    assert actual_order == required_order


def test_reset_requires_explicit_confirmation_flag():
    parser = reset_pricing_experiments.build_parser()

    args = parser.parse_args(["--yes"])

    assert args.yes is True
