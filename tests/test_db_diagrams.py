from pricing_pipeline.tools.db_diagrams import (
    ColumnInfo,
    DiagramSection,
    ForeignKeyInfo,
    SchemaMetadata,
    TableInfo,
    build_mermaid_er,
    build_overview_sections,
    prepare_display_metadata,
    render_html,
    _display_columns,
)


def _sample_metadata() -> SchemaMetadata:
    tables = [
        TableInfo(
            schema_name="pricing",
            table_name="PRICING_RATE_PACKAGE",
            columns=[
                ColumnInfo("rate_package_id", "bigint", False, True),
                ColumnInfo("model_name", "nvarchar(128)", False, False),
            ],
            row_count=3,
        ),
        TableInfo(
            schema_name="pricing",
            table_name="PRICING_TERM",
            columns=[
                ColumnInfo("term_id", "bigint", False, True),
                ColumnInfo("rate_package_id", "bigint", False, False, True),
                ColumnInfo("term_name", "nvarchar(128)", False, False),
            ],
            row_count=12,
        ),
    ]
    foreign_keys = [
        ForeignKeyInfo(
            fk_name="FK_TERM_PACKAGE",
            child_schema="pricing",
            child_table="PRICING_TERM",
            child_column="rate_package_id",
            parent_schema="pricing",
            parent_table="PRICING_RATE_PACKAGE",
            parent_column="rate_package_id",
        )
    ]
    return SchemaMetadata(tables=tables, foreign_keys=foreign_keys)


def _self_reference_metadata() -> SchemaMetadata:
    tables = [
        TableInfo(
            schema_name="pricing",
            table_name="PRICING_RATE_PACKAGE",
            columns=[
                ColumnInfo("rate_package_id", "bigint", False, True),
                ColumnInfo("parent_rate_package_id", "bigint", True, False, True),
                ColumnInfo("model_name", "nvarchar(128)", False, False),
            ],
            row_count=3,
        )
    ]
    foreign_keys = [
        ForeignKeyInfo(
            fk_name="FK_RATE_PACKAGE_PARENT",
            child_schema="pricing",
            child_table="PRICING_RATE_PACKAGE",
            child_column="parent_rate_package_id",
            parent_schema="pricing",
            parent_table="PRICING_RATE_PACKAGE",
            parent_column="rate_package_id",
        )
    ]
    return SchemaMetadata(tables=tables, foreign_keys=foreign_keys)


def _messy_metadata() -> SchemaMetadata:
    tables = [
        TableInfo("pricing", "PRICING_MODEL", [ColumnInfo("model_id", "bigint", False, True)], 2),
        TableInfo(
            "pricing",
            "MODEL_RUN",
            [
                ColumnInfo("model_run_id", "bigint", False, True),
                ColumnInfo("model_id", "bigint", False, False, True),
            ],
            4,
        ),
        TableInfo(
            "pricing",
            "PRICING_RATE_PACKAGE",
            [
                ColumnInfo("rate_package_id", "bigint", False, True),
                ColumnInfo("model_id", "bigint", False, False, True),
            ],
            3,
        ),
        TableInfo(
            "pricing",
            "PRICING_TERM",
            [
                ColumnInfo("term_id", "bigint", False, True),
                ColumnInfo("rate_package_id", "bigint", False, False, True),
            ],
            10,
        ),
        TableInfo(
            "pricing",
            "PRICING_RATE_CELL",
            [
                ColumnInfo("cell_id", "bigint", False, True),
                ColumnInfo("term_id", "bigint", False, False, True),
            ],
            20,
        ),
        TableInfo(
            "pricing",
            "PRICING_RATE_CELL_LEVEL",
            [
                ColumnInfo("cell_id", "bigint", False, True, True),
                ColumnInfo("feature_level_id", "bigint", False, False, True),
            ],
            20,
        ),
        TableInfo(
            "pricing",
            "PRICING_TERM_FEATURE",
            [
                ColumnInfo("term_id", "bigint", False, True, True),
                ColumnInfo("feature_id", "bigint", False, False, True),
                ColumnInfo("level_set_id", "bigint", False, False, True),
            ],
            10,
        ),
        TableInfo(
            "pricing",
            "PRICING_FEATURE",
            [ColumnInfo("feature_id", "bigint", False, True)],
            5,
        ),
        TableInfo(
            "pricing",
            "PRICING_FEATURE_LEVEL_SET",
            [
                ColumnInfo("level_set_id", "bigint", False, True),
                ColumnInfo("feature_id", "bigint", False, False, True),
            ],
            5,
        ),
        TableInfo(
            "pricing",
            "PRICING_FEATURE_LEVEL",
            [
                ColumnInfo("feature_level_id", "bigint", False, True),
                ColumnInfo("level_set_id", "bigint", False, False, True),
            ],
            25,
        ),
        TableInfo(
            "pricing",
            "PRICING_COMPILED_RATE_CELL",
            [
                ColumnInfo("rate_package_id", "bigint", False, True, True),
                ColumnInfo("term_id", "bigint", False, True, True),
                ColumnInfo("cell_key_digest", "varbinary(32)", False, True),
            ],
            20,
        ),
        TableInfo(
            "pricing",
            "PRICING_COMPILED_1D_RATE_BAND",
            [
                ColumnInfo("rate_package_id", "bigint", False, True, True),
                ColumnInfo("term_id", "bigint", False, True, True),
                ColumnInfo("feature_level_id", "bigint", False, True, True),
            ],
            20,
        ),
        TableInfo(
            "pricing",
            "PRICING_MODEL_DEPLOYMENT",
            [
                ColumnInfo("deployment_id", "bigint", False, True),
                ColumnInfo("model_id", "bigint", False, False, True),
                ColumnInfo("rate_package_id", "bigint", False, False, True),
            ],
            2,
        ),
        TableInfo(
            "pricing",
            "PRICING_PACKAGE_POINTER",
            [
                ColumnInfo("pointer_name", "nvarchar(128)", False, True),
                ColumnInfo("model_id", "bigint", True, False, True),
                ColumnInfo("rate_package_id", "bigint", False, False, True),
            ],
            2,
        ),
        TableInfo(
            "pricing",
            "DATASET_MANIFEST",
            [ColumnInfo("manifest_id", "nvarchar(128)", False, True)],
            1,
        ),
        TableInfo(
            "pricing",
            "DATASET_COLUMN",
            [
                ColumnInfo("manifest_id", "nvarchar(128)", False, True, True),
                ColumnInfo("ordinal_no", "int", False, True),
            ],
            8,
        ),
        TableInfo(
            "pricing",
            "FREMTPL_RAW",
            [ColumnInfo("IDpol", "bigint", False, True)],
            678_013,
        ),
        TableInfo(
            "pricing",
            "CV_SPLIT_SET",
            [
                ColumnInfo("split_set_id", "nvarchar(128)", False, True),
                ColumnInfo("manifest_id", "nvarchar(128)", False, False, True),
            ],
            1,
        ),
        TableInfo(
            "pricing",
            "CV_FOLD",
            [
                ColumnInfo("split_set_id", "nvarchar(128)", False, True, True),
                ColumnInfo("fold_no", "int", False, True),
            ],
            5,
        ),
        TableInfo(
            "pricing",
            "CV_FOLD_METRIC",
            [
                ColumnInfo("model_run_id", "bigint", False, True, True),
                ColumnInfo("split_set_id", "nvarchar(128)", False, True, True),
                ColumnInfo("fold_no", "int", False, True, True),
            ],
            10,
        ),
        TableInfo(
            "pricing_stg",
            "STG_RATE_CELL",
            [ColumnInfo("export_id", "nvarchar(128)", False, True)],
            100,
        ),
    ]
    foreign_keys = [
        ForeignKeyInfo(
            "FK_MODEL_RUN_MODEL",
            "pricing",
            "MODEL_RUN",
            "model_id",
            "pricing",
            "PRICING_MODEL",
            "model_id",
        ),
        ForeignKeyInfo(
            "FK_RATE_PACKAGE_MODEL",
            "pricing",
            "PRICING_RATE_PACKAGE",
            "model_id",
            "pricing",
            "PRICING_MODEL",
            "model_id",
        ),
        ForeignKeyInfo(
            "FK_MODEL_DEPLOYMENT_MODEL",
            "pricing",
            "PRICING_MODEL_DEPLOYMENT",
            "model_id",
            "pricing",
            "PRICING_MODEL",
            "model_id",
        ),
        ForeignKeyInfo(
            "FK_MODEL_DEPLOYMENT_PACKAGE",
            "pricing",
            "PRICING_MODEL_DEPLOYMENT",
            "rate_package_id",
            "pricing",
            "PRICING_RATE_PACKAGE",
            "rate_package_id",
        ),
        ForeignKeyInfo(
            "FK_PACKAGE_POINTER_MODEL",
            "pricing",
            "PRICING_PACKAGE_POINTER",
            "model_id",
            "pricing",
            "PRICING_MODEL",
            "model_id",
        ),
        ForeignKeyInfo(
            "FK_PACKAGE_POINTER_PACKAGE",
            "pricing",
            "PRICING_PACKAGE_POINTER",
            "rate_package_id",
            "pricing",
            "PRICING_RATE_PACKAGE",
            "rate_package_id",
        ),
        ForeignKeyInfo(
            "FK_TERM_PACKAGE",
            "pricing",
            "PRICING_TERM",
            "rate_package_id",
            "pricing",
            "PRICING_RATE_PACKAGE",
            "rate_package_id",
        ),
        ForeignKeyInfo(
            "FK_RATE_CELL_TERM",
            "pricing",
            "PRICING_RATE_CELL",
            "term_id",
            "pricing",
            "PRICING_TERM",
            "term_id",
        ),
        ForeignKeyInfo(
            "FK_RATE_CELL_LEVEL_CELL",
            "pricing",
            "PRICING_RATE_CELL_LEVEL",
            "cell_id",
            "pricing",
            "PRICING_RATE_CELL",
            "cell_id",
        ),
        ForeignKeyInfo(
            "FK_RATE_CELL_LEVEL_LEVEL",
            "pricing",
            "PRICING_RATE_CELL_LEVEL",
            "feature_level_id",
            "pricing",
            "PRICING_FEATURE_LEVEL",
            "feature_level_id",
        ),
        ForeignKeyInfo(
            "FK_TERM_FEATURE_TERM",
            "pricing",
            "PRICING_TERM_FEATURE",
            "term_id",
            "pricing",
            "PRICING_TERM",
            "term_id",
        ),
        ForeignKeyInfo(
            "FK_TERM_FEATURE_FEATURE",
            "pricing",
            "PRICING_TERM_FEATURE",
            "feature_id",
            "pricing",
            "PRICING_FEATURE",
            "feature_id",
        ),
        ForeignKeyInfo(
            "FK_TERM_FEATURE_LEVEL_SET",
            "pricing",
            "PRICING_TERM_FEATURE",
            "level_set_id",
            "pricing",
            "PRICING_FEATURE_LEVEL_SET",
            "level_set_id",
        ),
        ForeignKeyInfo(
            "FK_LEVEL_SET_FEATURE",
            "pricing",
            "PRICING_FEATURE_LEVEL_SET",
            "feature_id",
            "pricing",
            "PRICING_FEATURE",
            "feature_id",
        ),
        ForeignKeyInfo(
            "FK_FEATURE_LEVEL_SET",
            "pricing",
            "PRICING_FEATURE_LEVEL",
            "level_set_id",
            "pricing",
            "PRICING_FEATURE_LEVEL_SET",
            "level_set_id",
        ),
        ForeignKeyInfo(
            "FK_COMPILED_RATE_CELL_PACKAGE",
            "pricing",
            "PRICING_COMPILED_RATE_CELL",
            "rate_package_id",
            "pricing",
            "PRICING_RATE_PACKAGE",
            "rate_package_id",
        ),
        ForeignKeyInfo(
            "FK_COMPILED_RATE_CELL_TERM",
            "pricing",
            "PRICING_COMPILED_RATE_CELL",
            "term_id",
            "pricing",
            "PRICING_TERM",
            "term_id",
        ),
        ForeignKeyInfo(
            "FK_COMPILED_1D_RATE_BAND_PACKAGE",
            "pricing",
            "PRICING_COMPILED_1D_RATE_BAND",
            "rate_package_id",
            "pricing",
            "PRICING_RATE_PACKAGE",
            "rate_package_id",
        ),
        ForeignKeyInfo(
            "FK_COMPILED_1D_RATE_BAND_TERM",
            "pricing",
            "PRICING_COMPILED_1D_RATE_BAND",
            "term_id",
            "pricing",
            "PRICING_TERM",
            "term_id",
        ),
        ForeignKeyInfo(
            "FK_COMPILED_1D_RATE_BAND_LEVEL",
            "pricing",
            "PRICING_COMPILED_1D_RATE_BAND",
            "feature_level_id",
            "pricing",
            "PRICING_FEATURE_LEVEL",
            "feature_level_id",
        ),
        ForeignKeyInfo(
            "FK_DATASET_COLUMN_MANIFEST",
            "pricing",
            "DATASET_COLUMN",
            "manifest_id",
            "pricing",
            "DATASET_MANIFEST",
            "manifest_id",
        ),
        ForeignKeyInfo(
            "FK_MODEL_RUN_MANIFEST",
            "pricing",
            "MODEL_RUN",
            "manifest_id",
            "pricing",
            "DATASET_MANIFEST",
            "manifest_id",
        ),
        ForeignKeyInfo(
            "FK_CV_SPLIT_SET_MANIFEST",
            "pricing",
            "CV_SPLIT_SET",
            "manifest_id",
            "pricing",
            "DATASET_MANIFEST",
            "manifest_id",
        ),
        ForeignKeyInfo(
            "FK_CV_FOLD_SPLIT_SET",
            "pricing",
            "CV_FOLD",
            "split_set_id",
            "pricing",
            "CV_SPLIT_SET",
            "split_set_id",
        ),
        ForeignKeyInfo(
            "FK_CV_FOLD_METRIC_MODEL_RUN",
            "pricing",
            "CV_FOLD_METRIC",
            "model_run_id",
            "pricing",
            "MODEL_RUN",
            "model_run_id",
        ),
        ForeignKeyInfo(
            "FK_CV_FOLD_METRIC_FOLD",
            "pricing",
            "CV_FOLD_METRIC",
            "split_set_id",
            "pricing",
            "CV_FOLD",
            "split_set_id",
        ),
    ]
    return SchemaMetadata(tables=tables, foreign_keys=foreign_keys)


def test_build_mermaid_er_uses_foreign_key_relationships():
    mermaid = build_mermaid_er(_sample_metadata())

    assert "erDiagram" in mermaid
    assert "PRICING_RATE_PACKAGE ||--o{ PRICING_TERM : FK_TERM_PACKAGE" in mermaid
    assert "bigint rate_package_id PK" in mermaid
    assert "bigint rate_package_id FK" in mermaid


def test_self_referencing_foreign_keys_render_as_table_notes_not_loop_arrows():
    html = render_html(
        _self_reference_metadata(),
        database_name="PricingLab",
        schema_names=["pricing"],
    )

    assert "Self FK: parent_rate_package_id -> rate_package_id" in html
    assert "<title>FK_RATE_PACKAGE_PARENT</title>" not in html
    assert "||--o{ PRICING_RATE_PACKAGE : FK_RATE_PACKAGE_PARENT" not in html


def test_relationship_paths_are_orthogonal_not_sweeping_curves():
    html = render_html(_sample_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert '<path d="M ' in html
    assert " C " not in html
    assert " H " in html


def test_relationship_arrowheads_stop_outside_table_boxes():
    html = render_html(_sample_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert 'd="M 360 86 H 448"' in html
    assert 'd="M 344 86 H 464"' not in html


def test_render_html_is_self_contained_and_searchable():
    html = render_html(_sample_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert "<!doctype html>" in html
    assert "PricingLab ERD" in html
    assert "window.__diagramData" in html
    assert "PRICING_RATE_PACKAGE" in html
    assert "PRICING_TERM" in html
    assert "FK_TERM_PACKAGE" in html
    assert "Search tables, columns, or relationships" in html
    assert "https://" not in html


def test_render_html_includes_lineage_color_legend_and_section_classes():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert "Lineage Legend" in html
    assert "Full persisted data model" in html
    assert "Model/package lifecycle" in html
    assert "Rating lookup path" in html
    assert "Feature metadata" in html
    assert "Compiled outputs" in html
    assert "Dataset lineage" in html
    assert "CV audit" in html
    assert 'class="diagram-section lineage-full"' in html
    assert 'class="diagram-section lineage-model"' in html
    assert 'class="diagram-section lineage-rating"' in html
    assert 'class="diagram-section lineage-feature"' in html
    assert 'class="diagram-section lineage-compiled"' in html
    assert 'class="diagram-section lineage-dataset"' in html
    assert 'class="diagram-section lineage-cv"' in html


def test_relationship_arrows_and_tables_are_colored_by_lineage():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    full_html = html.split("<h2>Full persisted data model</h2>", 1)[1].split(
        "<h2>Core model lifecycle</h2>", 1
    )[0]

    assert 'class="relationship-edge edge-model"' in html
    assert 'class="relationship-edge edge-rating"' in html
    assert 'class="relationship-edge edge-feature"' in html
    assert 'class="relationship-edge edge-compiled"' in html
    assert 'class="relationship-edge edge-dataset"' in html
    assert 'class="relationship-edge edge-cv"' in html
    assert 'class="erd-table table-dataset"' in full_html
    assert 'class="erd-table table-model"' in full_html
    assert 'class="erd-table table-rating"' in html
    assert 'class="erd-table table-feature"' in html
    assert 'class="erd-table table-compiled"' in html
    assert 'class="erd-table table-cv"' in full_html


def test_pk_fk_badges_are_explicitly_colored_in_diagram_and_sidebar():
    html = render_html(_sample_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert 'class="svg-badge svg-badge-pk"' in html
    assert 'class="svg-badge svg-badge-fk"' in html
    assert '<span class="badge pk">PK</span>' in html
    assert '<span class="badge fk">FK</span>' in html


def test_pk_fk_badges_render_to_the_left_of_column_names():
    html = render_html(_sample_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert '<text x="47" y="76" text-anchor="middle">PK</text>' in html
    assert '<text x="88" y="78" fill="#e2e8f0" font-size="11">rate_package_id</text>' in html
    assert (
        '<td class="col-name"><span class="key-badges"><span class="badge pk">PK</span>'
        '</span>rate_package_id</td>'
    ) in html


def test_display_columns_put_primary_and_foreign_keys_before_descriptive_fields():
    table = TableInfo(
        "pricing",
        "EXAMPLE",
        [
            ColumnInfo("display_name", "nvarchar(128)", False, False),
            ColumnInfo("model_id", "bigint", False, False, True),
            ColumnInfo("example_id", "bigint", False, True),
            ColumnInfo("created_ts", "datetime2(3)", False, False),
        ],
        1,
    )

    assert [column.column_name for column in _display_columns(table)] == [
        "example_id",
        "model_id",
        "display_name",
        "created_ts",
    ]


def test_prepare_display_metadata_hides_staging_and_row_keys_by_default():
    metadata = prepare_display_metadata(_messy_metadata())
    table_names = {table.table_name for table in metadata.tables}

    assert "STG_RATE_CELL" not in table_names
    assert "PRICING_MODEL" in table_names
    assert "MODEL_RUN" in table_names


def test_render_html_splits_the_erd_into_focused_sections():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert "Full persisted data model" in html
    assert "Core model lifecycle" in html
    assert "Rating table structure" in html
    assert "Dataset lineage" in html
    assert "CV split audit" in html
    assert "STG_RATE_CELL" not in html
    assert "Hidden technical tables" in html
    assert "20 visible tables" in html


def test_overview_sections_use_domain_order_not_alphabetical_dump():
    sections = build_overview_sections(prepare_display_metadata(_messy_metadata()))
    full = next(section for section in sections if section.title == "Full persisted data model")
    core = next(section for section in sections if section.title == "Core model lifecycle")
    lineage = next(section for section in sections if section.title == "Dataset lineage")

    assert full.table_groups[0] == ["FREMTPL_RAW"]
    assert full.table_groups[1] == ["DATASET_MANIFEST", "DATASET_COLUMN"]
    assert full.table_groups[2] == ["CV_SPLIT_SET", "CV_FOLD"]
    assert full.table_groups[3] == ["PRICING_MODEL", "MODEL_RUN", "CV_FOLD_METRIC"]
    assert "STG_RATE_CELL" not in full.table_names
    assert isinstance(core, DiagramSection)
    assert core.table_groups[0] == ["PRICING_MODEL"]
    assert core.table_groups[1] == ["PRICING_RATE_PACKAGE"]
    assert core.table_groups[2] == ["PRICING_MODEL_DEPLOYMENT", "PRICING_PACKAGE_POINTER"]
    assert "MODEL_RUN" in lineage.table_names


def test_core_lifecycle_suppresses_redundant_model_edges_to_keep_layout_readable():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    core_html = html.split("<h2>Core model lifecycle</h2>", 1)[1].split(
        "<h2>Rating table structure</h2>", 1
    )[0]

    assert "<title>FK_MODEL_DEPLOYMENT_MODEL</title>" not in core_html
    assert "<title>FK_PACKAGE_POINTER_MODEL</title>" not in core_html
    assert "<title>FK_MODEL_DEPLOYMENT_PACKAGE</title>" in core_html
    assert "<title>FK_PACKAGE_POINTER_PACKAGE</title>" in core_html


def test_rating_structure_section_uses_readable_two_lane_layout():
    sections = build_overview_sections(prepare_display_metadata(_messy_metadata()))
    rating = next(section for section in sections if section.title == "Rating table structure")
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    rating_html = html.split("Rating table structure", 1)[1].split("Feature metadata", 1)[0]

    assert rating.table_groups == [
        ["PRICING_RATE_PACKAGE"],
        ["PRICING_TERM"],
        ["PRICING_RATE_CELL"],
        ["PRICING_RATE_CELL_LEVEL"],
        ["PRICING_FEATURE_LEVEL"],
    ]
    assert "PRICING_TERM_FEATURE" not in rating_html
    assert "PRICING_FEATURE_LEVEL_SET" not in rating_html
    assert ">PRICING_FEATURE</text>" not in rating_html
    assert "<title>FK_TERM_FEATURE_TERM</title>" not in rating_html
    assert "<title>FK_TERM_FEATURE_FEATURE</title>" not in rating_html
    assert "<title>FK_TERM_FEATURE_LEVEL_SET</title>" not in rating_html
    assert "<title>FK_LEVEL_SET_FEATURE</title>" not in rating_html


def test_feature_metadata_tables_have_their_own_section():
    sections = build_overview_sections(prepare_display_metadata(_messy_metadata()))
    feature_metadata = next(
        section for section in sections if section.title == "Feature metadata"
    )

    assert feature_metadata.table_groups == [
        ["PRICING_FEATURE"],
        ["PRICING_FEATURE_LEVEL_SET"],
        ["PRICING_FEATURE_LEVEL"],
        ["PRICING_TERM_FEATURE"],
    ]


def test_full_persisted_data_model_contains_visible_tables_once():
    sections = build_overview_sections(prepare_display_metadata(_messy_metadata()))
    full = next(section for section in sections if section.title == "Full persisted data model")
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    full_html = html.split("<h2>Full persisted data model</h2>", 1)[1].split(
        "<h2>Core model lifecycle</h2>", 1
    )[0]

    expected_tables = {
        "FREMTPL_RAW",
        "DATASET_MANIFEST",
        "DATASET_COLUMN",
        "PRICING_MODEL",
        "MODEL_RUN",
        "CV_SPLIT_SET",
        "CV_FOLD",
        "PRICING_RATE_PACKAGE",
        "PRICING_MODEL_DEPLOYMENT",
        "PRICING_PACKAGE_POINTER",
        "PRICING_TERM",
        "PRICING_RATE_CELL",
        "PRICING_TERM_FEATURE",
        "PRICING_FEATURE",
        "PRICING_FEATURE_LEVEL_SET",
        "PRICING_RATE_CELL_LEVEL",
        "PRICING_FEATURE_LEVEL",
        "PRICING_COMPILED_RATE_CELL",
        "PRICING_COMPILED_1D_RATE_BAND",
        "CV_FOLD_METRIC",
    }

    assert set(full.table_names) == expected_tables
    for table_name in expected_tables:
        assert f">{table_name}</text>" in full_html
    assert "STG_RATE_CELL" not in full_html


def test_full_persisted_data_model_is_split_into_workflow_lanes():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    full_html = html.split("<h2>Full persisted data model</h2>", 1)[1].split(
        "<h2>Core model lifecycle</h2>", 1
    )[0]

    assert 'class="workflow-lanes"' in full_html
    assert 'class="workflow-lane lineage-dataset"' in full_html
    assert 'class="workflow-lane lineage-cv"' in full_html
    assert 'class="workflow-lane lineage-model"' in full_html
    assert 'class="workflow-lane lineage-rating"' in full_html
    assert 'class="workflow-lane lineage-feature"' in full_html
    assert 'class="workflow-lane lineage-compiled"' in full_html
    assert "Dataset intake" in full_html
    assert "CV audit" in full_html
    assert "Model logging and publication" in full_html
    assert "Rating lookup path" in full_html
    assert "Feature dictionary" in full_html
    assert "Compiled outputs" in full_html


def test_full_persisted_data_model_draws_data_flow_not_every_fk():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    full_html = html.split("<h2>Full persisted data model</h2>", 1)[1].split(
        "<h2>Core model lifecycle</h2>", 1
    )[0]

    assert "FLOW_RAW_MANIFEST" in full_html
    assert "FLOW_RUN_PACKAGE" in full_html
    assert "ingest" in full_html
    assert "publish package" in full_html
    assert "<title>FK_MODEL_RUN_MANIFEST</title>" in full_html
    assert "<title>FK_MODEL_RUN_MODEL</title>" in full_html
    assert "<title>FK_TERM_PACKAGE</title>" in full_html
    assert "<title>FK_MODEL_DEPLOYMENT_MODEL</title>" not in full_html
    assert "<title>FK_PACKAGE_POINTER_MODEL</title>" not in full_html
    assert "<title>FK_COMPILED_RATE_CELL_TERM</title>" not in full_html
    assert "<title>FK_RATE_CELL_LEVEL_LEVEL</title>" not in full_html
    assert 'class="relationship-edge edge-dataset' in full_html
    assert 'class="relationship-edge edge-model' in full_html
    assert 'class="relationship-edge edge-rating' in full_html
    assert 'class="relationship-edge edge-feature' in full_html
    assert 'class="relationship-edge edge-compiled' in full_html
    assert 'class="relationship-edge edge-cv' in full_html


def test_svg_arrowheads_are_not_clipped_and_edges_are_labelled():
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])

    assert 'overflow="visible"' in html
    assert 'marker-end="url(#arrow-dataset)"' in html
    assert 'class="relationship-label label-dataset"' in html
    assert 'class="relationship-label label-model"' in html


def test_compiled_outputs_section_only_shows_package_to_output_tables():
    sections = build_overview_sections(prepare_display_metadata(_messy_metadata()))
    compiled = next(section for section in sections if section.title == "Compiled outputs")
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    compiled_html = html.split("<h2>Compiled outputs</h2>", 1)[1].split(
        "<h2>Dataset lineage</h2>", 1
    )[0]

    assert compiled.table_groups == [
        ["PRICING_RATE_PACKAGE"],
        ["PRICING_COMPILED_RATE_CELL", "PRICING_COMPILED_1D_RATE_BAND"],
    ]
    assert ">PRICING_TERM</text>" not in compiled_html
    assert "PRICING_FEATURE_LEVEL" not in compiled_html
    assert "<title>FK_COMPILED_RATE_CELL_PACKAGE</title>" in compiled_html
    assert "<title>FK_COMPILED_1D_RATE_BAND_PACKAGE</title>" in compiled_html
    assert "<title>FK_COMPILED_RATE_CELL_TERM</title>" not in compiled_html
    assert "<title>FK_COMPILED_1D_RATE_BAND_TERM</title>" not in compiled_html
    assert "<title>FK_COMPILED_1D_RATE_BAND_LEVEL</title>" not in compiled_html


def test_dataset_and_cv_lineage_are_split_into_small_sections():
    sections = build_overview_sections(prepare_display_metadata(_messy_metadata()))
    dataset = next(section for section in sections if section.title == "Dataset lineage")
    cv = next(section for section in sections if section.title == "CV split audit")
    html = render_html(_messy_metadata(), database_name="PricingLab", schema_names=["pricing"])
    dataset_html = html.split("<h2>Dataset lineage</h2>", 1)[1].split(
        "<h2>CV split audit</h2>", 1
    )[0]
    cv_html = html.split("<h2>CV split audit</h2>", 1)[1].split("Mermaid ER source", 1)[0]

    assert dataset.table_groups == [
        ["FREMTPL_RAW"],
        ["DATASET_MANIFEST"],
        ["DATASET_COLUMN", "MODEL_RUN"],
    ]
    assert cv.table_groups == [
        ["DATASET_MANIFEST"],
        ["CV_SPLIT_SET"],
        ["CV_FOLD"],
        ["CV_FOLD_METRIC"],
    ]
    assert "CV_FOLD_METRIC" not in dataset_html
    assert "FREMTPL_RAW" not in cv_html
    assert "<title>FK_CV_FOLD_METRIC_MODEL_RUN</title>" not in cv_html
