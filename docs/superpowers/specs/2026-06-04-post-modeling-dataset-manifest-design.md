# Post-Modeling Dataset Manifest Design

## Problem

Every model build needs dataset lineage, but the current demo makes users write a
small `DatasetSpec(...)` wrapper per model. The wrapper is mostly boilerplate,
yet the metadata it creates is important: without it the SQL catalog cannot
track which model-ready data produced a rate package.

The existing name `training_table` is also misleading. The final model trains on
the full model-ready data, and validation may happen separately. The lineage
manifest should describe the final model-ready source used for the package, not
necessarily an early raw/candidate training table.

## Goals

- Make dataset manifest housekeeping easy enough that users do not skip it.
- Keep SQL execution, source joins, and feature engineering in user-owned DAG
  tasks.
- Treat the post-modeling final model-ready source as the default manifest
  target.
- Keep TOML focused on stable metadata, not volatile SQL files or feature lists.
- Preserve the current publish/deploy separation.

## Non-Goals

- Do not put connection strings, auth, or Azure credential logic in TOML.
- Do not put SQL files or SQL execution instructions in `[dataset]`.
- Do not require final feature columns in TOML.
- Do not make the publish task infer or create dataset manifests implicitly.
- Do not change rate-package deployment semantics.

## Recommended Shape

`model.toml` owns stable dataset housekeeping:

```toml
[dataset]
dataset_name = "claim_frequency_modeling"
source_system = "policy_admin"
manifest_data_source = "pricing_work"
pk_columns = ["policy_id"]
target_column = "claim_count"
weight_column = "earned_exposure"
```

The user-owned ETL/modeling task owns SQL execution and materialization. After
feature selection/modeling, it returns a small payload pointing at the final
stable model-ready source:

```python
return {
    "modeling_schema": "pricing_work",
    "modeling_table": final_modeling_table,
    "rating_workbook_path": str(workbook_path),
    "model_version": model_version,
    "effective_from": effective_from,
    "export_id": export_id,
}
```

For relation-based sources, the library builds manifest SQL from the final
relation:

```sql
SELECT *
FROM pricing_work.<modeling_table>
ORDER BY policy_id
```

For edge cases, the user task may return `manifest_sql` directly:

```python
return {
    "manifest_sql": final_manifest_sql,
    ...
}
```

That escape hatch is for final model-ready queries only. SQL still lives in the
task or imported task code, not in TOML.

## DAG Flow

The production-style DAG default becomes:

```text
register model
-> pull/transform candidate data
-> train/select features/materialize final model-ready source
-> create manifest from final model-ready source
-> publish completed build
-> deploy separately after review
```

The publish task does not need dataset columns. It receives
`CompletedModelBuild` plus `manifest_id` / `split_set_id` from the manifest task.
Deployment remains a separate DAG/task that moves the model deployment slot.

## Components

### Dataset Config

Add an optional `DatasetBuildConfig` to the loaded model config:

- `dataset_name`
- `source_system`
- `manifest_data_source`
- `pk_columns`
- `target_column`
- `weight_column`

The config should validate required strings, non-empty PK columns, and optional
weight column. It should not contain `columns`, `sql_file`, or credentials.

### Dataset Spec Builder

Add a generic builder that converts:

- `DatasetBuildConfig`
- prepared/completed payload from the user task

into `DatasetSpec`.

Supported payload forms:

1. Relation form:

```python
{
    "modeling_schema": "pricing_work",
    "modeling_table": "CLAIM_FREQ_MODELING_RUN_123",
}
```

2. SQL form:

```python
{
    "manifest_sql": "SELECT ... ORDER BY policy_id",
}
```

The relation form should build `SELECT * FROM schema.table ORDER BY pk_columns`.
The SQL form should use the supplied SQL as-is.

### Manifest Task

Keep `create_prepared_dataset_manifest_task(...)` as the generic orchestration
task, but allow a standard dataset-builder factory so users do not hand-write a
per-model function:

```python
manifested = create_prepared_dataset_manifest_task(
    model_config=MODEL_CONFIG,
    dataset_builder=dataset_spec_from_model_config(MODEL_CONFIG),
)(completed_or_prepared)
```

The task adds `manifest_id` and `split_set_id` to the payload and returns the
enriched payload for `publish_completed_model_build_task(...)`.

### Naming

User-facing examples should prefer `modeling_table`, `modeling_schema`,
`modeling_dataset`, or `manifest_source`. Avoid `training_table` in new docs and
new demos except where it is existing compatibility code.

## Error Handling

The dataset builder should raise clear domain errors when:

- `[dataset]` is missing but the standard builder is used.
- Neither `manifest_sql` nor relation fields are supplied.
- Only one of `modeling_schema` / `modeling_table` is supplied.
- PK columns are missing or blank.
- Relation identifiers are blank.

It should not validate database connectivity. The manifest task already validates
the final SQL by executing it when creating `DATASET_MANIFEST` and
`DATASET_COLUMN`.

## Testing

Add focused tests for:

- TOML dataset config parsing.
- Relation payload builds expected `DatasetSpec.manifest_sql`.
- SQL payload passes explicit `manifest_sql` through.
- Missing relation/SQL payload raises a clear error.
- Manifest task can use the standard builder and pass the enriched payload to
  publish.
- Demo DAG uses post-modeling manifest naming in comments/docs.

Regression tests should confirm `publish_completed_model_build_task(...)` remains
unchanged: it should publish from supplied manifest IDs and should not deploy.

## Migration Impact

No SQL migration is required. This changes Python config/loading and DAG helper
API only. Existing tables and views continue to work.

Existing model folders may keep custom `dataset_builder` functions. The new
standard builder is a convenience path, not a forced replacement.
