# Portable underwriter report

The portable report is one Python file. Copy it into any project:

```bash
cp scripts/portable_underwriter_report.py ./
```

It does not import this repository, Airflow, SQL code, SuperGLM, Joblib, or
Node. The file contains PEP 723 metadata, so `uv` installs its four Python
dependencies—NumPy, Pandas, Plotly, and PyArrow—when it runs the file as a
command. Importing it from a notebook uses that notebook's environment; install
the same four packages there if they are not already available.

## Direct Python use

```python
from portable_underwriter_report import build_report

result = build_report(
    scored,
    actual="actual",
    predictions={"Current": "pred_current", "New": "pred_new"},
    sample_weight="exposure",
    features=["region", "vehicle_age"],
    model_type="frequency",
    output_path="model_review.html",
)

print(result.output_path)
```

`actual`, each prediction, `sample_weight`, and the optional
`comparison_unit` can be column names or one-dimensional arrays. The output is
one self-contained offline HTML file.

## Input scales

| `model_type` | Actual and predictions | `sample_weight` | Deviance |
|---|---|---|---|
| `frequency` | claim count / exposure | exposure | Poisson, `p=1` |
| `severity` | claim cost / claim count | claim count | Gamma, `p=2` |
| `burn_cost` | claim cost / exposure | exposure | Tweedie, default `p=1.5` |

Predictions must already include any model offset and be on the response-rate
scale. For burn cost, pass `tweedie_power=<value>` when the common reporting
power should differ from 1.5. The report does not estimate power or dispersion
from review outcomes.

## TOML use

Create `report.toml` beside the scored file:

```toml
[report]
output_path = "model_review.html"
title = "Pricing model review"
model_type = "burn_cost"
tweedie_power = 1.5
minimum_cell_size = 20

[data]
path = "scored.parquet"

[columns]
actual = "actual_burn_cost"
sample_weight = "exposure"
features = ["region", "vehicle_age"]
# comparison_unit = "policy_id"

[predictions]
"Current" = "pred_current"
"New" = "pred_new"
```

Then run:

```bash
uv run portable_underwriter_report.py --config report.toml
```

CSV, Feather, and Parquet inputs are supported. Relative data and output paths
are resolved from the TOML directory. Unknown sections, unknown options, and
mistyped values fail instead of being silently ignored.

## What scored-only mode provides

The basic call produces metrics, weighted prediction KDEs, local model-movement
heatmaps, Lorenz/gains curves, Gini, and configurable double lift with exposure
and paired comparison evidence.

Prediction columns cannot reveal isolated fitted main effects, EDF, confidence
intervals, interactions, or defensible feature importance. The script does not
invent them. Advanced integrations can import the neutral `ModelEvidence`,
`FeatureImportanceEvidence`, `MainEffectEvidence`, and `InteractionEvidence`
classes from the same file and pass an `evidence=` mapping. SuperGLM-specific
automatic extraction remains available in the full repository facade.

`minimum_cell_size` defaults to 20 distinct comparison units. Undersized chart
cells are coarsened or suppressed, and comparison-unit identifiers are never
written into the HTML. Do not also list an identifier column in `features`.

The executable tutorial is
[`tutorials/01_portable_underwriter_report.ipynb`](../../tutorials/01_portable_underwriter_report.ipynb).
