from __future__ import annotations

import json
from hashlib import sha256
import math
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from sqlalchemy import event, text

from pricing_pipeline.models.spec import ApprovedModelBuild, ApprovedModelBuildError


def _write_test_workbook(
    path: Path,
    *,
    core_timestamp: str,
    sheet_content: bytes,
) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as workbook:
        for member_name, content in (
            (
                "docProps/core.xml",
                (
                    "<cp:coreProperties xmlns:cp='urn:cp' xmlns:dcterms='urn:dcterms'>"
                    f"<dcterms:created>{core_timestamp}</dcterms:created>"
                    f"<dcterms:modified>{core_timestamp}</dcterms:modified>"
                    "</cp:coreProperties>"
                ).encode(),
            ),
            ("xl/worksheets/sheet1.xml", sheet_content),
        ):
            member = ZipInfo(member_name, date_time=(2026, 7, 18, 1, 2, 2))
            member.compress_type = ZIP_DEFLATED
            workbook.writestr(member, content)


def test_workbook_semantic_hash_ignores_generated_core_timestamps(tmp_path):
    from pricing_pipeline.workbench.submission import xlsx_semantic_sha256

    first = tmp_path / "first.xlsx"
    retry = tmp_path / "retry.xlsx"
    changed = tmp_path / "changed.xlsx"
    _write_test_workbook(
        first,
        core_timestamp="2026-07-18T00:00:01Z",
        sheet_content=b"<sheet>same rating table</sheet>",
    )
    _write_test_workbook(
        retry,
        core_timestamp="2026-07-18T00:00:02Z",
        sheet_content=b"<sheet>same rating table</sheet>",
    )
    _write_test_workbook(
        changed,
        core_timestamp="2026-07-18T00:00:02Z",
        sheet_content=b"<sheet>changed rating table</sheet>",
    )

    assert xlsx_semantic_sha256(first) == xlsx_semantic_sha256(retry)
    assert xlsx_semantic_sha256(first) != xlsx_semantic_sha256(changed)


def _model_spec(api):
    return api.PricingModelSpec(
        name="CLAIM_FREQUENCY",
        label="Claim frequency",
        target="claim_count",
        model_type="superglm_poisson",
        deployment_slot="CLAIM_FREQUENCY_UAT",
        features=("age",),
        dataset_name="claim_frequency_frame",
        source_system="pricing_sql",
        pk_columns=("policy_id",),
    )


def _local_model(api, tmp_path: Path):
    model_root = tmp_path / "pricing_models" / "claim_frequency"
    model_root.mkdir(parents=True)
    context = api.connect(mode="local", local_root=model_root / ".local")
    model = api.register_model(
        context,
        _model_spec(api),
        source_root=model_root,
        created_by="analyst@example.test",
    )
    return context, model


def _seed_lineage(
    context,
    *,
    manifest_id: str,
    split_set_id: str,
    artifact_uri: str,
) -> None:
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_MANIFEST (
                    manifest_id, dataset_name, source_system, data_as_of_date,
                    row_count, pk_columns_json, target_column, weight_column,
                    model_frame_sha256, frame_hash_metadata_json, created_by
                ) VALUES (
                    :manifest_id, 'claim_frequency_frame', 'pricing_sql',
                    '2026-06-30', 10, '["policy_id"]', 'claim_count', NULL,
                    :model_frame_sha256, :frame_hash_metadata_json, 'pytest'
                )
                """
            ),
            {
                "manifest_id": manifest_id,
                "model_frame_sha256": "f" * 64,
                "frame_hash_metadata_json": json.dumps(
                    {"frame_hash": {"format_version": 1}, "python": "test"},
                    sort_keys=True,
                ),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.DATASET_COLUMN (
                    manifest_id, ordinal_no, column_name, column_role,
                    pandas_dtype, null_count, distinct_count
                ) VALUES
                    (:manifest_id, 1, 'policy_id', 'KEY', 'int64', 0, 10),
                    (:manifest_id, 2, 'claim_count', 'TARGET', 'float64', 0, 4),
                    (:manifest_id, 3, 'age', 'FEATURE', 'float64', 0, 8)
                """
            ),
            {"manifest_id": manifest_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.CV_SPLIT_SET (
                    split_set_id, manifest_id, split_mode, splitter_class,
                    splitter_params_json, row_order_sha256, row_count,
                    fold_count, artifact_uri, artifact_sha256,
                    runtime_metadata_json, created_by
                ) VALUES (
                    :split_set_id, :manifest_id, 'MATERIALIZED',
                    'sklearn.model_selection.KFold', :splitter_params_json,
                    :row_order_sha256, 10, 2, :artifact_uri,
                    :artifact_sha256, :runtime_metadata_json, 'pytest'
                )
                """
            ),
            {
                "split_set_id": split_set_id,
                "manifest_id": manifest_id,
                "splitter_params_json": json.dumps(
                    {"n_splits": 2, "random_state": 42, "shuffle": True},
                    sort_keys=True,
                ),
                "row_order_sha256": "6" * 64,
                "artifact_uri": artifact_uri,
                "artifact_sha256": "9" * 64,
                "runtime_metadata_json": json.dumps(
                    {"python": "test", "packages": {"superglm": version("superglm")}},
                    sort_keys=True,
                ),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO pricing.CV_FOLD (
                    split_set_id, fold_no, n_train, n_test
                ) VALUES
                    (:split_set_id, 1, 6, 4),
                    (:split_set_id, 2, 6, 4)
                """
            ),
            {"split_set_id": split_set_id},
        )


def _complete_curve_points() -> tuple[dict[str, object], ...]:
    eta = math.log(1.2)
    points = []
    for split_no in (1, 2):
        points.extend(
            (
                {
                    "validation_split_no": split_no,
                    "term_name": "age_band",
                    "point_no": 1,
                    "point_kind": "LEVEL",
                    "x_numeric": None,
                    "level_text": "young",
                    "eta_contribution": 0.0,
                    "relativity": 1.0,
                    "support_value": 4.0,
                    "reference_value": None,
                    "reference_level": "young",
                },
                {
                    "validation_split_no": split_no,
                    "term_name": "age_band",
                    "point_no": 2,
                    "point_kind": "LEVEL",
                    "x_numeric": None,
                    "level_text": "old",
                    "eta_contribution": eta,
                    "relativity": math.exp(eta),
                    "support_value": 2.0,
                    "reference_value": None,
                    "reference_level": "young",
                },
            )
        )
    return tuple(points)


def _completed_build(
    tmp_path: Path,
    *,
    model,
    manifest_id: str,
    split_set_id: str,
    export_id: str = "build_" + "1" * 64,
    model_version: str = "v1",
    fingerprint: str = "1" * 64,
    curve_status: str = "COMPLETE",
) -> ApprovedModelBuild:
    attempt = (
        tmp_path
        / "pricing_models"
        / "claim_frequency"
        / ".local"
        / "workbench_artifacts"
        / model.name.lower()
        / model_version
        / export_id
    )
    attempt.mkdir(parents=True, exist_ok=True)
    workbook = attempt / "rating_tables.xlsx"
    workbook.write_bytes(b"canonical rating workbook")
    receipt = attempt / "publication_receipt.json"
    receipt.write_bytes(b'{"status":"complete"}')
    unavailable = curve_status == "UNAVAILABLE"
    validation_splits = (
        {
            "validation_split_no": 1,
            "n_train": 6,
            "n_validation": 4,
            "metrics": {"deviance": 1.1, "nll": 0.8, "gini": 0.31},
        },
        {
            "validation_split_no": 2,
            "n_train": 6,
            "n_validation": 4,
            "metrics": {"deviance": 1.3, "nll": 0.9, "gini": 0.29},
        },
    )
    return ApprovedModelBuild(
        model_id=model.model_id,
        model_name=model.name,
        rating_workbook_path=str(workbook),
        rating_workbook_sha256=sha256(workbook.read_bytes()).hexdigest(),
        model_version=model_version,
        model_type=model.config.model_type,
        target_name=model.config.target_name,
        deployment_slot=model.config.deployment_slot,
        effective_from="2026-07-01",
        export_id=export_id,
        manifest_id=manifest_id,
        split_set_id=split_set_id,
        created_by="analyst@example.test",
        mlflow_run_id="attempt-mlflow-run",
        publication_receipt_path=str(receipt),
        publication_receipt_sha256=sha256(receipt.read_bytes()).hexdigest(),
        candidate_artifact_path=str(attempt / "candidate.joblib"),
        candidate_artifact_sha256="c" * 64,
        candidate_artifact_format="superglm-candidate-joblib-v3",
        candidate_artifact_size_bytes=123,
        candidate_python_version=python_version(),
        candidate_superglm_version=version("superglm"),
        candidate_superglm_git_sha="e21bbdca98b6b511e189ae6c30f4af60ec09d95b",
        build_fingerprint_sha256=fingerprint,
        builder_source_sha256="2" * 64,
        materialized_split_sha256="3" * 64,
        runtime_sha256="4" * 64,
        candidate_superglm_sha256="5" * 64,
        row_order_sha256="6" * 64,
        model_source_sha256="e" * 64,
        model_frame_sha256="f" * 64,
        metrics={"cv_mean_deviance": 1.2},
        metric_scopes={"cv_mean_deviance": "validation_summary"},
        validation_splits=validation_splits,
        validation_curve_status=curve_status,
        validation_curve_reason=("curve capture failed" if unavailable else None),
        validation_curve_points=(() if unavailable else _complete_curve_points()),
    )


def _install_staging_stub(monkeypatch) -> None:
    from pricing_pipeline.publishing import sqlite_notebook

    def stage(engine, **kwargs):
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT OR REPLACE INTO pricing_stg.STG_RATING_EXPORT (
                        export_id, model_id, model_name, model_version,
                        base_rate, effective_from_date, source_file,
                        publication_receipt_json, publication_receipt_sha256,
                        package_metadata_json, offset_handling,
                        metadata_origin, staging_content_sha256, created_by
                    ) VALUES (
                        :export_id, :model_id, :model_name, :model_version,
                        0.25, :effective_from, :source_file,
                        '{}', :receipt_sha, :package_metadata_json, 'NONE',
                        'SUPERGLM_EXPORTER', :staging_digest, 'pytest'
                    )
                    """
                ),
                {
                    "export_id": kwargs["export_id"],
                    "model_id": kwargs["model_id"],
                    "model_name": kwargs["model_name"],
                    "model_version": kwargs["model_version"],
                    "effective_from": kwargs["effective_from"],
                    "source_file": str(kwargs["workbook_path"]),
                    "receipt_sha": kwargs["publication_receipt_sha256"],
                    "package_metadata_json": json.dumps(
                        {"model": {"family": "poisson", "link": "log"}},
                        sort_keys=True,
                    ),
                    "staging_digest": sha256(str(kwargs["workbook_path"]).encode()).hexdigest(),
                },
            )

    monkeypatch.setattr(sqlite_notebook, "stage_rating_export", stage)
    monkeypatch.setattr(
        sqlite_notebook,
        "_verify_candidate_artifact",
        lambda *args, **kwargs: None,
    )


def _publish_canonical_and_prepare_retry(api, context, model, tmp_path):
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    _seed_lineage(
        context,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
        artifact_uri=str(tmp_path / "canonical" / "splits.npz"),
    )
    canonical_build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=canonical_build.build_fingerprint_sha256,
        )
        == "v1"
    )
    canonical = api.publish_candidate(
        context,
        api.BuiltCandidate(model=model, completed_build=canonical_build),
    )
    _seed_lineage(
        context,
        manifest_id="manifest-retry",
        split_set_id="split-retry",
        artifact_uri=str(tmp_path / "retry" / "splits.npz"),
    )
    retry_build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-retry",
        split_set_id="split-retry",
        export_id="build_" + "a" * 64,
    )
    return canonical_build, canonical, retry_build


def test_local_root_publication_persists_complete_audit_evidence_and_queries_views(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _seed_lineage(
        context,
        manifest_id="manifest-attempt-1",
        split_set_id="split-attempt-1",
        artifact_uri=str(tmp_path / "attempt-1-splits.npz"),
    )
    _install_staging_stub(monkeypatch)
    build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-attempt-1",
        split_set_id="split-attempt-1",
    )

    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        == "v1"
    )
    curve_inserts: list[tuple[bool, int]] = []

    def capture_curve_insert(
        _connection,
        _cursor,
        statement,
        parameters,
        _context,
        executemany,
    ):
        if "INSERT INTO pricing.CV_SPLIT_CURVE_POINT" in statement:
            curve_inserts.append((executemany, len(parameters) if executemany else 1))
        return statement, parameters

    event.listen(context.engine, "before_cursor_execute", capture_curve_insert, retval=True)
    try:
        result = api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=build),
        )
    finally:
        event.remove(context.engine, "before_cursor_execute", capture_curve_insert)

    assert curve_inserts == [(True, 4)]
    assert result.model_version == "v1"
    assert result.manifest_id == "manifest-attempt-1"
    assert result.split_set_id == "split-attempt-1"
    assert result.was_existing is False
    with context.engine.connect() as connection:
        package = (
            connection.execute(
                text(
                    """
                SELECT build_fingerprint_sha256
                FROM pricing.PRICING_RATE_PACKAGE
                WHERE rate_package_id = :rate_package_id
                """
                ),
                {"rate_package_id": result.rate_package_id},
            )
            .mappings()
            .one()
        )
        model_run = (
            connection.execute(
                text(
                    """
                SELECT
                    model_run_id, validation_source_model_run_id,
                    builder_source_sha256, materialized_split_sha256,
                    runtime_sha256, candidate_superglm_sha256,
                    candidate_python_version, candidate_superglm_version,
                    candidate_superglm_git_sha, validation_curve_status,
                    validation_curve_reason
                FROM pricing.MODEL_RUN
                WHERE model_run_id = :model_run_id
                """
                ),
                {"model_run_id": result.model_run_id},
            )
            .mappings()
            .one()
        )
        run_metrics = connection.execute(
            text(
                """
                SELECT metric_name, metric_value, metric_scope
                FROM mlops.MODEL_RUN_METRIC
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": result.model_run_id},
        ).all()
        split_rows = connection.execute(
            text(
                """
                SELECT validation_split_no, n_train, n_validation,
                       deviance, nll, gini
                FROM pricing.V_MODEL_VALIDATION_SPLIT
                WHERE model_run_id = :model_run_id
                ORDER BY validation_split_no
                """
            ),
            {"model_run_id": result.model_run_id},
        ).all()
        summary_rows = connection.execute(
            text(
                """
                SELECT validation_split_count, mean_deviance,
                       validation_curve_status
                FROM pricing.V_MODEL_VALIDATION_SUMMARY
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": result.model_run_id},
        ).all()
        curve_rows = connection.execute(
            text(
                """
                SELECT validation_split_no, term_name, point_no,
                       model_fit_scope
                FROM pricing.V_MODEL_VALIDATION_SPLIT_RELATIVITY
                WHERE model_run_id = :model_run_id
                ORDER BY validation_split_no, term_name, point_no
                """
            ),
            {"model_run_id": result.model_run_id},
        ).all()
        current_split_rows = connection.execute(
            text(
                """
                SELECT validation_split_no, n_train, n_validation
                FROM pricing.V_CURRENT_DATASET_VALIDATION_SPLIT
                WHERE manifest_id = :manifest_id
                ORDER BY validation_split_no
                """
            ),
            {"manifest_id": result.manifest_id},
        ).all()
        final_rows = connection.execute(
            text(
                """
                SELECT model_fit_scope
                FROM pricing.V_FINAL_MODEL_RELATIVITY
                WHERE rate_package_id = :rate_package_id
                """
            ),
            {"rate_package_id": result.rate_package_id},
        ).all()

    assert package["build_fingerprint_sha256"] == "1" * 64
    assert model_run == {
        "model_run_id": str(result.model_run_id),
        "validation_source_model_run_id": str(result.model_run_id),
        "builder_source_sha256": "2" * 64,
        "materialized_split_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "candidate_superglm_sha256": "5" * 64,
        "candidate_python_version": build.candidate_python_version,
        "candidate_superglm_version": build.candidate_superglm_version,
        "candidate_superglm_git_sha": build.candidate_superglm_git_sha,
        "validation_curve_status": "COMPLETE",
        "validation_curve_reason": None,
    }
    assert run_metrics == [("cv_mean_deviance", 1.2, "validation_summary")]
    assert split_rows == [
        (1, 6, 4, 1.1, 0.8, 0.31),
        (2, 6, 4, 1.3, 0.9, 0.29),
    ]
    assert summary_rows == [(2, pytest.approx(1.2), "COMPLETE")]
    assert curve_rows == [
        (split_no, "age_band", point_no, "VALIDATION_TRAINING_SPLIT_MODEL")
        for split_no in (1, 2)
        for point_no in (1, 2)
    ]
    assert current_split_rows == [(1, 6, 4), (2, 6, 4)]
    assert final_rows == []


def test_identical_fingerprint_reuses_canonical_root_across_attempt_identifiers_and_paths(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _seed_lineage(
        context,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
        artifact_uri=str(tmp_path / "canonical" / "splits.npz"),
    )
    canonical_build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=canonical_build.build_fingerprint_sha256,
        )
        == "v1"
    )
    canonical = api.publish_candidate(
        context,
        api.BuiltCandidate(model=model, completed_build=canonical_build),
    )

    _seed_lineage(
        context,
        manifest_id="manifest-retry",
        split_set_id="split-retry",
        artifact_uri=str(tmp_path / "retry" / "different-splits.npz"),
    )
    retry_build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-retry",
        split_set_id="split-retry",
        export_id="different-attempt-export-id",
        model_version="v1",
    ).model_copy(
        update={
            "candidate_artifact_sha256": "d" * 64,
            "candidate_artifact_size_bytes": 456,
            "mlflow_run_id": "different-attempt-mlflow-run",
            "created_by": "retrying.analyst@example.test",
        }
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=retry_build.build_fingerprint_sha256,
        )
        == "v1"
    )

    reused = api.publish_candidate(
        context,
        api.BuiltCandidate(model=model, completed_build=retry_build),
    )

    assert reused.was_existing is True
    assert reused.rate_package_id == canonical.rate_package_id
    assert reused.model_run_id == canonical.model_run_id
    assert reused.model_version == canonical.model_version
    assert reused.manifest_id == canonical.manifest_id
    assert reused.split_set_id == canonical.split_set_id
    assert reused.export_id == canonical.export_id
    assert reused.rating_workbook_path == canonical.rating_workbook_path
    assert reused.mlflow_run_id == canonical.mlflow_run_id
    with context.engine.connect() as connection:
        package_count = connection.execute(
            text("SELECT COUNT(*) FROM pricing.PRICING_RATE_PACKAGE")
        ).scalar_one()
        run_count = connection.execute(text("SELECT COUNT(*) FROM pricing.MODEL_RUN")).scalar_one()
        linked_attempt = connection.execute(
            text(
                """
                SELECT manifest_id, split_set_id
                FROM pricing.MODEL_RUN
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": reused.model_run_id},
        ).one()
    assert package_count == 1
    assert run_count == 1
    assert linked_attempt == ("manifest-canonical", "split-canonical")


@pytest.mark.parametrize(
    ("field_name", "artifact_label"),
    (
        ("rating_workbook_path", "rating workbook"),
        ("publication_receipt_path", "publication receipt"),
    ),
)
def test_incoming_workbook_and_receipt_must_be_inside_artifact_root_before_staging(
    monkeypatch,
    tmp_path,
    field_name,
    artifact_label,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _seed_lineage(
        context,
        manifest_id="manifest-incoming",
        split_set_id="split-incoming",
        artifact_uri=str(tmp_path / "incoming" / "splits.npz"),
    )
    build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-incoming",
        split_set_id="split-incoming",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        == "v1"
    )
    source = Path(getattr(build, field_name))
    outside = tmp_path / "outside-artifact-root" / source.name
    outside.parent.mkdir()
    outside.write_bytes(source.read_bytes())
    outside_build = build.model_copy(update={field_name: str(outside)})

    with pytest.raises(
        ApprovedModelBuildError,
        match=rf"{artifact_label}.*outside.*artifact root",
    ):
        api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=outside_build),
        )

    with context.engine.connect() as connection:
        counts = (
            connection.execute(
                text("SELECT COUNT(*) FROM pricing_stg.STG_RATING_EXPORT")
            ).scalar_one(),
            connection.execute(
                text("SELECT COUNT(*) FROM pricing.PRICING_RATE_PACKAGE")
            ).scalar_one(),
        )
    assert counts == (0, 0)


@pytest.mark.parametrize(
    ("damage", "error_fragment"),
    (("missing", "missing"), ("corrupt", "SHA-256")),
)
def test_incoming_receipt_is_verified_before_staging(
    monkeypatch,
    tmp_path,
    damage,
    error_fragment,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _seed_lineage(
        context,
        manifest_id="manifest-incoming",
        split_set_id="split-incoming",
        artifact_uri=str(tmp_path / "incoming" / "splits.npz"),
    )
    build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-incoming",
        split_set_id="split-incoming",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        == "v1"
    )
    receipt = Path(build.publication_receipt_path)
    if damage == "missing":
        receipt.unlink()
    else:
        receipt.write_bytes(b"corrupt receipt")

    with pytest.raises(
        ApprovedModelBuildError,
        match=rf"publication receipt.*{error_fragment}",
    ):
        api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=build),
        )

    with context.engine.connect() as connection:
        counts = (
            connection.execute(
                text("SELECT COUNT(*) FROM pricing_stg.STG_RATING_EXPORT")
            ).scalar_one(),
            connection.execute(
                text("SELECT COUNT(*) FROM pricing.PRICING_RATE_PACKAGE")
            ).scalar_one(),
        )
    assert counts == (0, 0)


@pytest.mark.parametrize(
    ("artifact_kind", "damage", "error_fragment"),
    (
        ("workbook", "missing", "missing"),
        ("workbook", "corrupt", "SHA-256"),
        ("workbook", "outside", "outside"),
        ("receipt", "missing", "missing"),
        ("receipt", "corrupt", "SHA-256"),
        ("receipt", "outside", "outside"),
    ),
)
def test_retry_verifies_canonical_workbook_and_receipt_durability(
    monkeypatch,
    tmp_path,
    artifact_kind,
    damage,
    error_fragment,
):
    from pricing_pipeline import notebook as api

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    canonical_build, canonical, retry_build = _publish_canonical_and_prepare_retry(
        api,
        context,
        model,
        tmp_path,
    )
    canonical_path = Path(
        canonical_build.rating_workbook_path
        if artifact_kind == "workbook"
        else canonical_build.publication_receipt_path
    )
    if damage == "missing":
        canonical_path.unlink()
    elif damage == "corrupt":
        canonical_path.write_bytes(b"corrupt canonical artifact")
    else:
        outside = tmp_path / "outside-canonical-artifacts" / canonical_path.name
        outside.parent.mkdir()
        outside.write_bytes(canonical_path.read_bytes())
        with context.engine.begin() as connection:
            if artifact_kind == "workbook":
                connection.execute(
                    text(
                        """
                        UPDATE pricing.PRICING_RATE_PACKAGE
                        SET rating_workbook_path = :path
                        WHERE rate_package_id = :rate_package_id
                        """
                    ),
                    {"path": str(outside), "rate_package_id": canonical.rate_package_id},
                )
                connection.execute(
                    text(
                        """
                        UPDATE pricing.MODEL_RUN
                        SET rating_workbook_path = :path
                        WHERE model_run_id = :model_run_id
                        """
                    ),
                    {"path": str(outside), "model_run_id": canonical.model_run_id},
                )
            else:
                connection.execute(
                    text(
                        """
                        UPDATE pricing.MODEL_RUN
                        SET publication_receipt_path = :path
                        WHERE model_run_id = :model_run_id
                        """
                    ),
                    {"path": str(outside), "model_run_id": canonical.model_run_id},
                )

    artifact_label = "rating workbook" if artifact_kind == "workbook" else "publication receipt"
    with pytest.raises(
        ApprovedModelBuildError,
        match=rf"canonical {artifact_label}.*{error_fragment}",
    ):
        api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=retry_build),
        )

    assert Path(retry_build.rating_workbook_path).is_file()
    assert Path(retry_build.publication_receipt_path).is_file()
    with context.engine.connect() as connection:
        counts = (
            connection.execute(
                text("SELECT COUNT(*) FROM pricing.PRICING_RATE_PACKAGE")
            ).scalar_one(),
            connection.execute(text("SELECT COUNT(*) FROM pricing.MODEL_RUN")).scalar_one(),
        )
    assert counts == (1, 1)


@pytest.mark.parametrize(
    "corruption",
    ("move_fold", "duplicate_fold", "move_curve", "duplicate_curve"),
)
def test_retry_rejects_fold_and_curve_rows_owned_by_another_split_set(
    monkeypatch,
    tmp_path,
    corruption,
):
    from pricing_pipeline import notebook as api

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _, canonical, retry_build = _publish_canonical_and_prepare_retry(
        api,
        context,
        model,
        tmp_path,
    )
    _seed_lineage(
        context,
        manifest_id="manifest-foreign",
        split_set_id="split-foreign",
        artifact_uri=str(tmp_path / "foreign" / "splits.npz"),
    )
    with context.engine.begin() as connection:
        if corruption == "move_fold":
            connection.execute(
                text(
                    """
                    UPDATE pricing.CV_FOLD_METRIC
                    SET split_set_id = 'split-foreign'
                    WHERE model_run_id = :model_run_id
                      AND fold_no = 1
                      AND metric_name = 'deviance'
                    """
                ),
                {"model_run_id": canonical.model_run_id},
            )
        elif corruption == "duplicate_fold":
            connection.execute(
                text(
                    """
                    INSERT INTO pricing.CV_FOLD_METRIC (
                        model_run_id, split_set_id, fold_no,
                        metric_name, metric_value
                    )
                    SELECT model_run_id, 'split-foreign', fold_no,
                           metric_name, metric_value
                    FROM pricing.CV_FOLD_METRIC
                    WHERE model_run_id = :model_run_id
                      AND split_set_id = 'split-canonical'
                      AND fold_no = 1
                      AND metric_name = 'deviance'
                    """
                ),
                {"model_run_id": canonical.model_run_id},
            )
        elif corruption == "move_curve":
            connection.execute(
                text(
                    """
                    UPDATE pricing.CV_SPLIT_CURVE_POINT
                    SET split_set_id = 'split-foreign'
                    WHERE model_run_id = :model_run_id
                      AND split_no = 1
                      AND term_name = 'age_band'
                      AND point_no = 1
                    """
                ),
                {"model_run_id": canonical.model_run_id},
            )
        else:
            connection.execute(
                text(
                    """
                    INSERT INTO pricing.CV_SPLIT_CURVE_POINT (
                        model_run_id, split_set_id, split_no,
                        term_name, point_no, point_kind,
                        x_numeric, level_text, eta_contribution,
                        relativity, support_value, reference_value,
                        reference_level
                    )
                    SELECT model_run_id, 'split-foreign', split_no,
                           term_name, point_no, point_kind,
                           x_numeric, level_text, eta_contribution,
                           relativity, support_value, reference_value,
                           reference_level
                    FROM pricing.CV_SPLIT_CURVE_POINT
                    WHERE model_run_id = :model_run_id
                      AND split_set_id = 'split-canonical'
                      AND split_no = 1
                      AND term_name = 'age_band'
                      AND point_no = 1
                    """
                ),
                {"model_run_id": canonical.model_run_id},
            )

    with pytest.raises(ValueError, match="incompatible model-run evidence"):
        api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=retry_build),
        )


def test_material_change_allocates_new_version_and_unavailable_curve_has_zero_points(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    for suffix in ("baseline", "changed"):
        _seed_lineage(
            context,
            manifest_id=f"manifest-{suffix}",
            split_set_id=f"split-{suffix}",
            artifact_uri=str(tmp_path / suffix / "splits.npz"),
        )

    baseline_build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-baseline",
        split_set_id="split-baseline",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=baseline_build.build_fingerprint_sha256,
        )
        == "v1"
    )
    baseline = api.publish_candidate(
        context,
        api.BuiltCandidate(model=model, completed_build=baseline_build),
    )

    changed_build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-changed",
        split_set_id="split-changed",
        export_id="build_" + "7" * 64,
        model_version="v2",
        fingerprint="7" * 64,
        curve_status="UNAVAILABLE",
    ).model_copy(update={"model_source_sha256": "a" * 64})
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=changed_build.build_fingerprint_sha256,
        )
        == "v2"
    )
    changed = api.publish_candidate(
        context,
        api.BuiltCandidate(model=model, completed_build=changed_build),
    )

    assert baseline.model_version == "v1"
    assert changed.model_version == "v2"
    assert changed.rate_package_id != baseline.rate_package_id
    with context.engine.connect() as connection:
        roots = connection.execute(
            text(
                """
                SELECT model_version, build_fingerprint_sha256
                FROM pricing.PRICING_RATE_PACKAGE
                ORDER BY rate_package_id
                """
            )
        ).all()
        curve_capture = connection.execute(
            text(
                """
                SELECT validation_curve_status, validation_curve_reason,
                       validation_source_model_run_id
                FROM pricing.MODEL_RUN
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": changed.model_run_id},
        ).one()
        point_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM pricing.CV_SPLIT_CURVE_POINT
                WHERE model_run_id = :model_run_id
                """
            ),
            {"model_run_id": changed.model_run_id},
        ).scalar_one()
    assert roots == [("v1", "1" * 64), ("v2", "7" * 64)]
    assert curve_capture == (
        "UNAVAILABLE",
        "curve capture failed",
        str(changed.model_run_id),
    )
    assert point_count == 0


def test_version_resolution_rejects_canonical_root_reservation_disagreement(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _seed_lineage(
        context,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
        artifact_uri=str(tmp_path / "canonical" / "splits.npz"),
    )
    build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        == "v1"
    )
    api.publish_candidate(context, api.BuiltCandidate(model=model, completed_build=build))
    with context.engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE pricing.PRICING_MODEL_VERSION_RESERVATION
                SET model_version = 'v9'
                WHERE model_id = :model_id
                  AND export_id = :reservation_export_id
                """
            ),
            {
                "model_id": model.model_id,
                "reservation_export_id": f"build_{build.build_fingerprint_sha256}",
            },
        )

    with pytest.raises(RuntimeError, match="canonical root.*reservation"):
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )


def test_same_fingerprint_rejects_changed_semantic_evidence_without_new_rows(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _seed_lineage(
        context,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
        artifact_uri=str(tmp_path / "canonical" / "splits.npz"),
    )
    build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-canonical",
        split_set_id="split-canonical",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        == "v1"
    )
    api.publish_candidate(context, api.BuiltCandidate(model=model, completed_build=build))

    conflicting = build.model_copy(
        update={
            "builder_source_sha256": "8" * 64,
            "metrics": {"cv_mean_deviance": 99.0},
        }
    )
    with pytest.raises(ValueError, match="incompatible model-run evidence"):
        api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=conflicting),
        )

    geometry_payload = build.model_dump()
    geometry_payload["validation_splits"][0]["n_train"] = 7
    conflicting_geometry = ApprovedModelBuild(**geometry_payload)
    with pytest.raises(ValueError, match="validation split geometry"):
        api.publish_candidate(
            context,
            api.BuiltCandidate(model=model, completed_build=conflicting_geometry),
        )

    with context.engine.connect() as connection:
        counts = [
            connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table in (
                "pricing.PRICING_RATE_PACKAGE",
                "pricing.MODEL_RUN",
                "mlops.MODEL_RUN_METRIC",
                "pricing.CV_SPLIT_CURVE_POINT",
            )
        ]
    assert counts == [1, 1, 1, 4]


def test_curve_insert_failure_rolls_back_package_run_metrics_and_points(
    monkeypatch,
    tmp_path,
):
    from pricing_pipeline import notebook as api
    from pricing_pipeline.publishing.sqlite_notebook import (
        resolve_sqlite_model_version,
    )

    context, model = _local_model(api, tmp_path)
    _install_staging_stub(monkeypatch)
    _seed_lineage(
        context,
        manifest_id="manifest-atomic",
        split_set_id="split-atomic",
        artifact_uri=str(tmp_path / "atomic" / "splits.npz"),
    )
    build = _completed_build(
        tmp_path,
        model=model,
        manifest_id="manifest-atomic",
        split_set_id="split-atomic",
    )
    assert (
        resolve_sqlite_model_version(
            context.engine,
            model_name=model.name,
            build_fingerprint_sha256=build.build_fingerprint_sha256,
        )
        == "v1"
    )

    def fail_curve_insert(
        _connection,
        _cursor,
        statement,
        parameters,
        _context,
        _executemany,
    ):
        if "INSERT INTO pricing.CV_SPLIT_CURVE_POINT" in statement:
            raise RuntimeError("synthetic curve write failure")
        return statement, parameters

    event.listen(context.engine, "before_cursor_execute", fail_curve_insert, retval=True)
    try:
        with pytest.raises(RuntimeError, match="synthetic curve write failure"):
            api.publish_candidate(
                context,
                api.BuiltCandidate(model=model, completed_build=build),
            )
    finally:
        event.remove(context.engine, "before_cursor_execute", fail_curve_insert)

    with context.engine.connect() as connection:
        counts = [
            connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table in (
                "pricing.PRICING_RATE_PACKAGE",
                "pricing.MODEL_RUN",
                "mlops.MODEL_RUN_DATASET",
                "mlops.MODEL_RUN_SPLIT_SET",
                "mlops.MODEL_RUN_METRIC",
                "pricing.CV_FOLD_METRIC",
                "pricing.CV_SPLIT_CURVE_POINT",
            )
        ]
    assert counts == [0, 0, 0, 0, 0, 0, 0]
