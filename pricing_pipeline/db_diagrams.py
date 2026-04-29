from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class ColumnInfo:
    column_name: str
    data_type: str
    nullable: bool
    primary_key: bool
    foreign_key: bool = False


@dataclass(frozen=True)
class TableInfo:
    schema_name: str
    table_name: str
    columns: list[ColumnInfo]
    row_count: int

    @property
    def key(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def diagram_id(self) -> str:
        return _diagram_identifier(self.schema_name, self.table_name)


@dataclass(frozen=True)
class ForeignKeyInfo:
    fk_name: str
    child_schema: str
    child_table: str
    child_column: str
    parent_schema: str
    parent_table: str
    parent_column: str

    @property
    def child_key(self) -> str:
        return f"{self.child_schema}.{self.child_table}"

    @property
    def parent_key(self) -> str:
        return f"{self.parent_schema}.{self.parent_table}"


@dataclass(frozen=True)
class DiagramLogicalEdge:
    edge_name: str
    source_table: str
    target_table: str
    label: str
    lineage: str


@dataclass(frozen=True)
class DiagramLane:
    title: str
    description: str
    table_groups: list[list[str]]
    lineage: str
    drawn_fk_names: tuple[str, ...] = ()
    logical_edges: tuple[DiagramLogicalEdge, ...] = ()


@dataclass(frozen=True)
class SchemaMetadata:
    tables: list[TableInfo]
    foreign_keys: list[ForeignKeyInfo]


@dataclass(frozen=True)
class DiagramSection:
    title: str
    description: str
    table_groups: list[list[str]]
    lineage: str = "other"
    hidden_fk_names: tuple[str, ...] = ()
    drawn_fk_names: tuple[str, ...] = ()
    logical_edges: tuple[DiagramLogicalEdge, ...] = ()
    table_lineages: tuple[tuple[str, str], ...] = ()
    fk_lineages: tuple[tuple[str, str], ...] = ()
    fk_labels: tuple[tuple[str, str], ...] = ()
    lanes: tuple[DiagramLane, ...] = ()

    @property
    def table_names(self) -> list[str]:
        return [table_name for group in self.table_groups for table_name in group]


FULL_TABLE_LINEAGES: tuple[tuple[str, str], ...] = (
    ("FREMTPL_RAW", "dataset"),
    ("DATASET_MANIFEST", "dataset"),
    ("DATASET_COLUMN", "dataset"),
    ("CV_SPLIT_SET", "cv"),
    ("CV_FOLD", "cv"),
    ("CV_FOLD_METRIC", "cv"),
    ("PRICING_MODEL", "model"),
    ("MODEL_RUN", "model"),
    ("PRICING_MODEL_DEPLOYMENT", "model"),
    ("PRICING_PACKAGE_POINTER", "model"),
    ("PRICING_RATE_PACKAGE", "rating"),
    ("PRICING_TERM", "rating"),
    ("PRICING_RATE_CELL", "rating"),
    ("PRICING_RATE_CELL_LEVEL", "rating"),
    ("PRICING_TERM_FEATURE", "feature"),
    ("PRICING_FEATURE", "feature"),
    ("PRICING_FEATURE_LEVEL_SET", "feature"),
    ("PRICING_FEATURE_LEVEL", "feature"),
    ("PRICING_COMPILED_RATE_CELL", "compiled"),
    ("PRICING_COMPILED_1D_RATE_BAND", "compiled"),
)

FULL_FLOW_FK_LINEAGES: tuple[tuple[str, str], ...] = (
    ("FK_DATASET_COLUMN_MANIFEST", "dataset"),
    ("FK_CV_SPLIT_SET_MANIFEST", "cv"),
    ("FK_CV_FOLD_SPLIT_SET", "cv"),
    ("FK_CV_FOLD_METRIC_FOLD", "cv"),
    ("FK_CV_FOLD_METRIC_MODEL_RUN", "cv"),
    ("FK_MODEL_RUN_MANIFEST", "model"),
    ("FK_MODEL_RUN_MODEL", "model"),
    ("FK_TERM_PACKAGE", "rating"),
    ("FK_RATE_CELL_TERM", "rating"),
    ("FK_RATE_CELL_LEVEL_CELL", "rating"),
    ("FK_RATE_CELL_LEVEL_LEVEL", "feature"),
    ("FK_TERM_FEATURE_TERM", "feature"),
    ("FK_TERM_FEATURE_FEATURE", "feature"),
    ("FK_TERM_FEATURE_LEVEL_SET", "feature"),
    ("FK_LEVEL_SET_FEATURE", "feature"),
    ("FK_FEATURE_LEVEL_SET", "feature"),
    ("FK_MODEL_DEPLOYMENT_PACKAGE", "model"),
    ("FK_PACKAGE_POINTER_PACKAGE", "model"),
    ("FK_COMPILED_RATE_CELL_PACKAGE", "compiled"),
    ("FK_COMPILED_1D_RATE_BAND_PACKAGE", "compiled"),
)

FULL_FLOW_FK_LABELS: tuple[tuple[str, str], ...] = (
    ("FK_DATASET_COLUMN_MANIFEST", "columns"),
    ("FK_CV_SPLIT_SET_MANIFEST", "split set"),
    ("FK_CV_FOLD_SPLIT_SET", "fold rows"),
    ("FK_CV_FOLD_METRIC_FOLD", "fold metrics"),
    ("FK_CV_FOLD_METRIC_MODEL_RUN", "run metrics"),
    ("FK_MODEL_RUN_MANIFEST", "training data"),
    ("FK_MODEL_RUN_MODEL", "model run"),
    ("FK_TERM_PACKAGE", "terms"),
    ("FK_RATE_CELL_TERM", "cells"),
    ("FK_RATE_CELL_LEVEL_CELL", "cell levels"),
    ("FK_RATE_CELL_LEVEL_LEVEL", "rated level"),
    ("FK_TERM_FEATURE_TERM", "term inputs"),
    ("FK_TERM_FEATURE_FEATURE", "feature"),
    ("FK_TERM_FEATURE_LEVEL_SET", "level set"),
    ("FK_LEVEL_SET_FEATURE", "level sets"),
    ("FK_FEATURE_LEVEL_SET", "levels"),
    ("FK_MODEL_DEPLOYMENT_PACKAGE", "deployed package"),
    ("FK_PACKAGE_POINTER_PACKAGE", "active pointer"),
    ("FK_COMPILED_RATE_CELL_PACKAGE", "compiled cells"),
    ("FK_COMPILED_1D_RATE_BAND_PACKAGE", "compiled bands"),
)

FULL_LOGICAL_EDGES: tuple[DiagramLogicalEdge, ...] = (
    DiagramLogicalEdge("FLOW_RAW_MANIFEST", "FREMTPL_RAW", "DATASET_MANIFEST", "ingest", "dataset"),
    DiagramLogicalEdge(
        "FLOW_RUN_PACKAGE",
        "MODEL_RUN",
        "PRICING_RATE_PACKAGE",
        "publish package",
        "model",
    ),
)

FULL_WORKFLOW_LANES: tuple[DiagramLane, ...] = (
    DiagramLane(
        title="Dataset intake",
        description="Raw FremTPL rows are captured as a versioned dataset manifest and column profile.",
        table_groups=[
            ["FREMTPL_RAW"],
            ["DATASET_MANIFEST"],
            ["DATASET_COLUMN"],
        ],
        lineage="dataset",
        drawn_fk_names=("FK_DATASET_COLUMN_MANIFEST",),
        logical_edges=(
            DiagramLogicalEdge(
                "FLOW_RAW_MANIFEST",
                "FREMTPL_RAW",
                "DATASET_MANIFEST",
                "ingest",
                "dataset",
            ),
        ),
    ),
    DiagramLane(
        title="CV audit",
        description="Replayable split metadata links dataset rows, folds, model runs, and fold metrics.",
        table_groups=[
            ["DATASET_MANIFEST"],
            ["CV_SPLIT_SET"],
            ["CV_FOLD"],
            ["MODEL_RUN", "CV_FOLD_METRIC"],
        ],
        lineage="cv",
        drawn_fk_names=(
            "FK_CV_SPLIT_SET_MANIFEST",
            "FK_CV_FOLD_SPLIT_SET",
            "FK_CV_FOLD_METRIC_FOLD",
            "FK_CV_FOLD_METRIC_MODEL_RUN",
        ),
    ),
    DiagramLane(
        title="Model logging and publication",
        description="Airflow training logs a model run, then publishes a rate package and deployment pointer.",
        table_groups=[
            ["DATASET_MANIFEST", "PRICING_MODEL"],
            ["MODEL_RUN"],
            ["PRICING_RATE_PACKAGE"],
            ["PRICING_MODEL_DEPLOYMENT", "PRICING_PACKAGE_POINTER"],
        ],
        lineage="model",
        drawn_fk_names=(
            "FK_MODEL_RUN_MANIFEST",
            "FK_MODEL_RUN_MODEL",
            "FK_MODEL_DEPLOYMENT_PACKAGE",
            "FK_PACKAGE_POINTER_PACKAGE",
        ),
        logical_edges=(
            DiagramLogicalEdge(
                "FLOW_RUN_PACKAGE",
                "MODEL_RUN",
                "PRICING_RATE_PACKAGE",
                "publish package",
                "model",
            ),
        ),
    ),
    DiagramLane(
        title="Rating lookup path",
        description="The published package resolves terms to cells and cell-level combinations.",
        table_groups=[
            ["PRICING_RATE_PACKAGE"],
            ["PRICING_TERM"],
            ["PRICING_RATE_CELL"],
            ["PRICING_RATE_CELL_LEVEL"],
        ],
        lineage="rating",
        drawn_fk_names=(
            "FK_TERM_PACKAGE",
            "FK_RATE_CELL_TERM",
            "FK_RATE_CELL_LEVEL_CELL",
        ),
    ),
    DiagramLane(
        title="Feature dictionary",
        description="Reusable feature definitions, level sets, concrete levels, and term-feature mappings.",
        table_groups=[
            ["PRICING_FEATURE"],
            ["PRICING_FEATURE_LEVEL_SET"],
            ["PRICING_FEATURE_LEVEL", "PRICING_TERM_FEATURE"],
        ],
        lineage="feature",
        drawn_fk_names=(
            "FK_LEVEL_SET_FEATURE",
            "FK_FEATURE_LEVEL_SET",
            "FK_TERM_FEATURE_FEATURE",
            "FK_TERM_FEATURE_LEVEL_SET",
        ),
    ),
    DiagramLane(
        title="Compiled outputs",
        description="Read-optimized rating outputs for downstream inspection and lookup.",
        table_groups=[
            ["PRICING_RATE_PACKAGE"],
            ["PRICING_COMPILED_RATE_CELL", "PRICING_COMPILED_1D_RATE_BAND"],
        ],
        lineage="compiled",
        drawn_fk_names=(
            "FK_COMPILED_RATE_CELL_PACKAGE",
            "FK_COMPILED_1D_RATE_BAND_PACKAGE",
        ),
    ),
)


def load_schema_metadata(engine: Engine, schema_names: Iterable[str]) -> SchemaMetadata:
    schemas = tuple(dict.fromkeys(schema_names))
    if not schemas:
        raise ValueError("At least one SQL Server schema must be provided.")

    with engine.connect() as con:
        column_rows = list(
            con.execute(
                _COLUMN_SQL.bindparams(bindparam("schema_names", expanding=True)),
                {"schema_names": schemas},
            ).mappings()
        )
        fk_rows = list(
            con.execute(
                _FOREIGN_KEY_SQL.bindparams(bindparam("schema_names", expanding=True)),
                {"schema_names": schemas},
            ).mappings()
        )

        tables_by_key: dict[str, dict[str, object]] = {}
        for row in column_rows:
            key = f"{row['schema_name']}.{row['table_name']}"
            table = tables_by_key.setdefault(
                key,
                {
                    "schema_name": row["schema_name"],
                    "table_name": row["table_name"],
                    "row_count": int(row["row_count"] or 0),
                    "columns": [],
                },
            )
            table["columns"].append(
                ColumnInfo(
                    column_name=row["column_name"],
                    data_type=_format_data_type(
                        row["type_name"],
                        row["max_length"],
                        row["precision"],
                        row["scale"],
                    ),
                    nullable=bool(row["is_nullable"]),
                    primary_key=bool(row["is_primary_key"]),
                    foreign_key=bool(row["is_foreign_key"]),
                )
            )

        tables = [
            TableInfo(
                schema_name=str(table["schema_name"]),
                table_name=str(table["table_name"]),
                columns=list(table["columns"]),
                row_count=int(table["row_count"]),
            )
            for table in tables_by_key.values()
        ]
        tables.sort(key=lambda table: (table.schema_name.lower(), table.table_name.lower()))

        foreign_keys = [
            ForeignKeyInfo(
                fk_name=row["fk_name"],
                child_schema=row["child_schema"],
                child_table=row["child_table"],
                child_column=row["child_column"],
                parent_schema=row["parent_schema"],
                parent_table=row["parent_table"],
                parent_column=row["parent_column"],
            )
            for row in fk_rows
        ]
        foreign_keys.sort(
            key=lambda fk: (
                fk.child_schema.lower(),
                fk.child_table.lower(),
                fk.fk_name.lower(),
                fk.child_column.lower(),
            )
        )

    return SchemaMetadata(tables=tables, foreign_keys=foreign_keys)


def prepare_display_metadata(
    metadata: SchemaMetadata,
    *,
    include_staging: bool = False,
    include_row_keys: bool = False,
) -> SchemaMetadata:
    tables = [
        table
        for table in metadata.tables
        if _include_table(
            table,
            include_staging=include_staging,
            include_row_keys=include_row_keys,
        )
    ]
    visible_keys = {table.key for table in tables}
    foreign_keys = [
        fk for fk in metadata.foreign_keys if fk.child_key in visible_keys and fk.parent_key in visible_keys
    ]
    return SchemaMetadata(tables=tables, foreign_keys=foreign_keys)


def build_overview_sections(metadata: SchemaMetadata) -> list[DiagramSection]:
    existing = {table.table_name for table in metadata.tables}
    sections = [
        DiagramSection(
            title="Full persisted data model",
            description=(
                "Dataset-to-model workflow map across persisted tables; staging and row-key "
                "materialization remain hidden by default."
            ),
            table_groups=[
                ["FREMTPL_RAW"],
                ["DATASET_MANIFEST", "DATASET_COLUMN"],
                ["CV_SPLIT_SET", "CV_FOLD"],
                ["PRICING_MODEL", "MODEL_RUN", "CV_FOLD_METRIC"],
                [
                    "PRICING_RATE_PACKAGE",
                    "PRICING_MODEL_DEPLOYMENT",
                    "PRICING_PACKAGE_POINTER",
                ],
                [
                    "PRICING_TERM",
                    "PRICING_RATE_CELL",
                    "PRICING_TERM_FEATURE",
                ],
                [
                    "PRICING_FEATURE",
                    "PRICING_FEATURE_LEVEL_SET",
                    "PRICING_FEATURE_LEVEL",
                ],
                [
                    "PRICING_RATE_CELL_LEVEL",
                    "PRICING_COMPILED_RATE_CELL",
                    "PRICING_COMPILED_1D_RATE_BAND",
                ],
            ],
            lineage="full",
            drawn_fk_names=tuple(name for name, _lineage in FULL_FLOW_FK_LINEAGES),
            logical_edges=FULL_LOGICAL_EDGES,
            table_lineages=FULL_TABLE_LINEAGES,
            fk_lineages=FULL_FLOW_FK_LINEAGES,
            fk_labels=FULL_FLOW_FK_LABELS,
            lanes=FULL_WORKFLOW_LANES,
        ),
        DiagramSection(
            title="Core model lifecycle",
            description="Model families, training runs, published packages, and deployment pointers.",
            table_groups=[
                ["PRICING_MODEL"],
                ["PRICING_RATE_PACKAGE"],
                ["PRICING_MODEL_DEPLOYMENT", "PRICING_PACKAGE_POINTER"],
            ],
            hidden_fk_names=(
                "FK_MODEL_DEPLOYMENT_MODEL",
                "FK_PACKAGE_POINTER_MODEL",
            ),
            lineage="model",
        ),
        DiagramSection(
            title="Rating table structure",
            description="The core lookup path from a published package to the cells and feature levels it rates.",
            table_groups=[
                ["PRICING_RATE_PACKAGE"],
                ["PRICING_TERM"],
                ["PRICING_RATE_CELL"],
                ["PRICING_RATE_CELL_LEVEL"],
                ["PRICING_FEATURE_LEVEL"],
            ],
            lineage="rating",
        ),
        DiagramSection(
            title="Feature metadata",
            description="Support tables that describe reusable features, level sets, and term-feature mappings.",
            table_groups=[
                ["PRICING_FEATURE"],
                ["PRICING_FEATURE_LEVEL_SET"],
                ["PRICING_FEATURE_LEVEL"],
                ["PRICING_TERM_FEATURE"],
            ],
            lineage="feature",
        ),
        DiagramSection(
            title="Compiled outputs",
            description="Read-optimized output tables produced from a published rating package.",
            table_groups=[
                ["PRICING_RATE_PACKAGE"],
                ["PRICING_COMPILED_RATE_CELL", "PRICING_COMPILED_1D_RATE_BAND"],
            ],
            hidden_fk_names=(
                "FK_COMPILED_RATE_CELL_TERM",
                "FK_COMPILED_1D_RATE_BAND_TERM",
                "FK_COMPILED_1D_RATE_BAND_LEVEL",
            ),
            lineage="compiled",
        ),
        DiagramSection(
            title="Dataset lineage",
            description="Raw source table, dataset manifest, captured columns, and training runs.",
            table_groups=[
                ["FREMTPL_RAW"],
                ["DATASET_MANIFEST"],
                ["DATASET_COLUMN", "MODEL_RUN"],
            ],
            lineage="dataset",
        ),
        DiagramSection(
            title="CV split audit",
            description="Replayable split definitions, materialized folds, and per-fold metrics.",
            table_groups=[
                ["DATASET_MANIFEST"],
                ["CV_SPLIT_SET"],
                ["CV_FOLD"],
                ["CV_FOLD_METRIC"],
            ],
            hidden_fk_names=("FK_CV_FOLD_METRIC_MODEL_RUN",),
            lineage="cv",
        ),
    ]

    focused_sections: list[DiagramSection] = []
    section_tables: set[str] = set()
    for section in sections:
        groups = [[name for name in group if name in existing] for group in section.table_groups]
        groups = [group for group in groups if group]
        if groups:
            focused_sections.append(
                DiagramSection(
                    title=section.title,
                    description=section.description,
                    table_groups=groups,
                    lineage=section.lineage,
                    hidden_fk_names=section.hidden_fk_names,
                    drawn_fk_names=section.drawn_fk_names,
                    logical_edges=section.logical_edges,
                    table_lineages=section.table_lineages,
                    fk_lineages=section.fk_lineages,
                    fk_labels=section.fk_labels,
                    lanes=section.lanes,
                )
            )
            section_tables.update(name for group in groups for name in group)

    remaining = sorted(existing - section_tables)
    if remaining:
        focused_sections.append(
            DiagramSection(
                title="Other persisted tables",
                description="Additional persisted objects that are not part of the main pricing path.",
                table_groups=[remaining],
            )
        )

    return focused_sections


def build_mermaid_er(metadata: SchemaMetadata) -> str:
    lines = ["erDiagram"]
    for table in metadata.tables:
        lines.append(f"    {table.diagram_id} {{")
        for column in _display_columns(table):
            flags = []
            if column.primary_key:
                flags.append("PK")
            if column.foreign_key:
                flags.append("FK")
            suffix = f" {' '.join(flags)}" if flags else ""
            lines.append(f"        {_mermaid_type(column.data_type)} {column.column_name}{suffix}")
        lines.append("    }")

    seen_relationships: set[tuple[str, str, str]] = set()
    for fk in metadata.foreign_keys:
        if _is_self_fk(fk):
            continue
        parent_id = _diagram_identifier(fk.parent_schema, fk.parent_table)
        child_id = _diagram_identifier(fk.child_schema, fk.child_table)
        relationship = (parent_id, child_id, fk.fk_name)
        if relationship in seen_relationships:
            continue
        seen_relationships.add(relationship)
        lines.append(f"    {parent_id} ||--o{{ {child_id} : {fk.fk_name}")

    return "\n".join(lines) + "\n"


def _display_columns(table: TableInfo) -> list[ColumnInfo]:
    original_order = {id(column): index for index, column in enumerate(table.columns)}
    return sorted(
        table.columns,
        key=lambda column: (
            not column.primary_key,
            not column.foreign_key,
            original_order[id(column)],
        ),
    )


def render_html(
    metadata: SchemaMetadata,
    *,
    database_name: str,
    schema_names: Iterable[str],
    include_staging: bool = False,
    include_row_keys: bool = False,
) -> str:
    schemas = list(schema_names)
    display_metadata = prepare_display_metadata(
        metadata,
        include_staging=include_staging,
        include_row_keys=include_row_keys,
    )
    sections = build_overview_sections(display_metadata)
    hidden_count = len(metadata.tables) - len(display_metadata.tables)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    mermaid = build_mermaid_er(display_metadata)
    diagram_sections = _render_diagram_sections(display_metadata, sections)
    diagram_data = _metadata_json(display_metadata)
    schema_label = ", ".join(schemas)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(database_name)} ERD</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111827;
      --panel: #172033;
      --panel-2: #1f2937;
      --ink: #f8fafc;
      --muted: #94a3b8;
      --line: #334155;
      --accent: #38bdf8;
      --accent-2: #a7f3d0;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 22px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: #0f172a;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 26px;
      line-height: 1.15;
      font-weight: 720;
    }}
    .subhead {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #111827;
      color: var(--muted);
      white-space: nowrap;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .legend-title {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      align-self: center;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 26px;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #111827;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .legend-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--lineage);
      box-shadow: 0 0 0 3px var(--lineage-soft);
    }}
    .lineage-full {{ --lineage: #cbd5e1; --lineage-soft: rgba(203, 213, 225, 0.14); }}
    .lineage-model {{ --lineage: #60a5fa; --lineage-soft: rgba(96, 165, 250, 0.18); }}
    .lineage-rating {{ --lineage: #34d399; --lineage-soft: rgba(52, 211, 153, 0.18); }}
    .lineage-feature {{ --lineage: #fbbf24; --lineage-soft: rgba(251, 191, 36, 0.18); }}
    .lineage-compiled {{ --lineage: #a78bfa; --lineage-soft: rgba(167, 139, 250, 0.18); }}
    .lineage-dataset {{ --lineage: #2dd4bf; --lineage-soft: rgba(45, 212, 191, 0.18); }}
    .lineage-cv {{ --lineage: #22d3ee; --lineage-soft: rgba(34, 211, 238, 0.18); }}
    .lineage-other {{ --lineage: #94a3b8; --lineage-soft: rgba(148, 163, 184, 0.16); }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      min-height: calc(100vh - 93px);
    }}
    .diagram-pane {{
      min-width: 0;
      padding: 18px;
      overflow: auto;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 3;
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(17, 24, 39, 0.94);
      backdrop-filter: blur(8px);
    }}
    input[type="search"] {{
      width: min(540px, 100%);
      min-height: 38px;
      padding: 8px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f172a;
      color: var(--ink);
      font: inherit;
    }}
    button {{
      min-height: 38px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #1e293b;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    .canvas {{
      width: max-content;
      min-width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #0b1120;
      overflow: hidden;
    }}
    .diagram-section {{
      margin-bottom: 18px;
    }}
    .workflow-lanes {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
    }}
    .workflow-lane {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(15, 23, 42, 0.58);
      overflow: hidden;
    }}
    .workflow-lane-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--lineage-soft);
    }}
    .workflow-lane-header h3 {{
      margin: 0;
      font-size: 14px;
      line-height: 1.2;
    }}
    .workflow-lane-header p {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .section-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px 14px;
      margin: 0 0 8px;
    }}
    .section-header h2 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }}
    .section-header p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: #0f172a;
      overflow: auto;
      padding: 18px;
    }}
    aside h2 {{
      margin: 0 0 12px;
      font-size: 15px;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 680;
    }}
    .table-card {{
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      overflow: hidden;
    }}
    .table-card[hidden] {{ display: none; }}
    .table-card header {{
      padding: 10px 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }}
    .table-card h3 {{
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      word-break: break-word;
    }}
    .table-card .meta {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .columns {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .columns td {{
      padding: 6px 8px;
      border-bottom: 1px solid rgba(51, 65, 85, 0.65);
      vertical-align: top;
    }}
    .columns tr:last-child td {{ border-bottom: 0; }}
    .col-name {{
      color: var(--ink);
      word-break: break-word;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .col-type {{ color: var(--muted); text-align: right; white-space: nowrap; }}
    .relationship-note {{
      padding: 7px 8px;
      border-top: 1px solid rgba(51, 65, 85, 0.65);
      color: var(--accent-2);
      font-size: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 740;
    }}
    .key-badges {{
      display: inline-flex;
      flex: 0 0 50px;
      gap: 4px;
      min-width: 50px;
    }}
    .pk {{ background: rgba(56, 189, 248, 0.16); color: var(--accent); }}
    .fk {{ background: rgba(167, 243, 208, 0.14); color: var(--accent-2); }}
    details {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #0f172a;
    }}
    summary {{
      cursor: pointer;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    pre {{
      margin: 0;
      overflow: auto;
      padding: 12px;
      border-top: 1px solid var(--line);
      color: #cbd5e1;
      font-size: 12px;
    }}
    .erd-table.filtered {{ opacity: 0.16; }}
    .relationship-edge {{
      fill: none;
      stroke-width: 1.8;
      opacity: 0.92;
    }}
    .relationship-edge.is-logical {{
      stroke-dasharray: 6 5;
    }}
    .relationship-label {{
      font-size: 10px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #0b1120;
      stroke-width: 4px;
      stroke-linejoin: round;
    }}
    .edge-model {{ stroke: #60a5fa; }}
    .edge-full {{ stroke: #cbd5e1; }}
    .edge-rating {{ stroke: #34d399; }}
    .edge-feature {{ stroke: #fbbf24; }}
    .edge-compiled {{ stroke: #a78bfa; }}
    .edge-dataset {{ stroke: #2dd4bf; }}
    .edge-cv {{ stroke: #22d3ee; }}
    .edge-other {{ stroke: #94a3b8; }}
    .label-model {{ fill: #bfdbfe; }}
    .label-full {{ fill: #e2e8f0; }}
    .label-rating {{ fill: #bbf7d0; }}
    .label-feature {{ fill: #fde68a; }}
    .label-compiled {{ fill: #ddd6fe; }}
    .label-dataset {{ fill: #99f6e4; }}
    .label-cv {{ fill: #a5f3fc; }}
    .label-other {{ fill: #cbd5e1; }}
    .table-shell {{ fill: #172033; stroke-width: 1.2; }}
    .table-head {{ stroke-width: 1.2; }}
    .table-model .table-shell, .table-model .table-head {{ stroke: #60a5fa; }}
    .table-full .table-shell, .table-full .table-head {{ stroke: #cbd5e1; }}
    .table-rating .table-shell, .table-rating .table-head {{ stroke: #34d399; }}
    .table-feature .table-shell, .table-feature .table-head {{ stroke: #fbbf24; }}
    .table-compiled .table-shell, .table-compiled .table-head {{ stroke: #a78bfa; }}
    .table-dataset .table-shell, .table-dataset .table-head {{ stroke: #2dd4bf; }}
    .table-cv .table-shell, .table-cv .table-head {{ stroke: #22d3ee; }}
    .table-other .table-shell, .table-other .table-head {{ stroke: #94a3b8; }}
    .table-model .table-head {{ fill: rgba(96, 165, 250, 0.16); }}
    .table-full .table-head {{ fill: rgba(203, 213, 225, 0.12); }}
    .table-rating .table-head {{ fill: rgba(52, 211, 153, 0.15); }}
    .table-feature .table-head {{ fill: rgba(251, 191, 36, 0.16); }}
    .table-compiled .table-head {{ fill: rgba(167, 139, 250, 0.16); }}
    .table-dataset .table-head {{ fill: rgba(45, 212, 191, 0.15); }}
    .table-cv .table-head {{ fill: rgba(34, 211, 238, 0.15); }}
    .table-other .table-head {{ fill: #1f2937; }}
    .svg-badge rect {{ rx: 4; }}
    .svg-badge text {{ font-size: 9px; font-weight: 800; }}
    .svg-badge-pk rect {{ fill: rgba(96, 165, 250, 0.2); stroke: #60a5fa; }}
    .svg-badge-pk text {{ fill: #bfdbfe; }}
    .svg-badge-fk rect {{ fill: rgba(52, 211, 153, 0.18); stroke: #34d399; }}
    .svg-badge-fk text {{ fill: #bbf7d0; }}
    @media (max-width: 1050px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(database_name)} ERD</h1>
    <div class="subhead">
      <span class="pill">Schemas: {html.escape(schema_label)}</span>
      <span class="pill">{len(display_metadata.tables)} visible tables</span>
      <span class="pill">{len(display_metadata.foreign_keys)} visible FK columns</span>
      <span class="pill">Hidden technical tables: {hidden_count}</span>
      <span class="pill">Generated {generated_at}</span>
    </div>
    {_render_lineage_legend()}
  </header>
  <main>
    <section class="diagram-pane" aria-label="Database relationship diagram">
      <div class="toolbar">
        <input id="search" type="search" placeholder="Search tables, columns, or relationships" autofocus>
        <button id="reset" type="button">Reset</button>
      </div>
      {diagram_sections}
      <details>
        <summary>Mermaid ER source</summary>
        <pre>{html.escape(mermaid)}</pre>
      </details>
    </section>
    <aside aria-label="Table details">
      <h2>Tables</h2>
      {_render_table_cards(display_metadata)}
    </aside>
  </main>
  <script>
    window.__diagramData = {diagram_data};
    const search = document.querySelector("#search");
    const reset = document.querySelector("#reset");
    const tableCards = Array.from(document.querySelectorAll(".table-card"));
    const tableGroups = Array.from(document.querySelectorAll(".erd-table"));

    function applyFilter() {{
      const q = search.value.trim().toLowerCase();
      tableCards.forEach(card => {{
        const match = !q || card.dataset.search.includes(q);
        card.hidden = !match;
      }});
      tableGroups.forEach(group => {{
        const match = !q || group.dataset.search.includes(q);
        group.classList.toggle("filtered", !match);
      }});
    }}

    search.addEventListener("input", applyFilter);
    reset.addEventListener("click", () => {{
      search.value = "";
      applyFilter();
      search.focus();
    }});
  </script>
</body>
</html>
"""


def write_diagram_site(
    metadata: SchemaMetadata,
    *,
    output_dir: Path,
    database_name: str,
    schema_names: Iterable[str],
    include_staging: bool = False,
    include_row_keys: bool = False,
) -> None:
    display_metadata = prepare_display_metadata(
        metadata,
        include_staging=include_staging,
        include_row_keys=include_row_keys,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(
        render_html(
            metadata,
            database_name=database_name,
            schema_names=schema_names,
            include_staging=include_staging,
            include_row_keys=include_row_keys,
        ),
        encoding="utf-8",
    )
    (output_dir / "schema.mmd").write_text(
        build_mermaid_er(display_metadata),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(_metadata_dict(display_metadata), indent=2),
        encoding="utf-8",
    )


def _include_table(
    table: TableInfo,
    *,
    include_staging: bool,
    include_row_keys: bool,
) -> bool:
    table_name = table.table_name.upper()
    if not include_staging and table_name.startswith("STG_"):
        return False
    if not include_row_keys and table_name in {"DATASET_ROW_KEY", "STG_DATASET_ROW_KEY"}:
        return False
    return True


def _render_lineage_legend() -> str:
    items = [
        ("full", "Full persisted model"),
        ("model", "Model/package lifecycle"),
        ("rating", "Rating lookup path"),
        ("feature", "Feature metadata"),
        ("compiled", "Compiled outputs"),
        ("dataset", "Dataset lineage"),
        ("cv", "CV audit"),
    ]
    rendered_items = "\n".join(
        f"""      <span class="legend-item lineage-{lineage}"><span class="legend-swatch"></span>{label}</span>"""
        for lineage, label in items
    )
    return f"""<div class="legend" aria-label="Lineage Legend">
      <span class="legend-title">Lineage Legend</span>
{rendered_items}
    </div>"""


def _render_svg_markers() -> str:
    colors = {
        "full": "#cbd5e1",
        "model": "#60a5fa",
        "rating": "#34d399",
        "feature": "#fbbf24",
        "compiled": "#a78bfa",
        "dataset": "#2dd4bf",
        "cv": "#22d3ee",
        "other": "#94a3b8",
    }
    return "\n    ".join(
        f"""<marker id="arrow-{lineage}" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth" overflow="visible">
      <path d="M 0 0 L 10 4 L 0 8 z" fill="{color}"></path>
    </marker>"""
        for lineage, color in colors.items()
    )


def _render_svg_badges(x: int, y: int, badges: list[str]) -> str:
    rendered = []
    current_x = x
    for badge in badges:
        badge_class = "svg-badge-pk" if badge == "PK" else "svg-badge-fk"
        rendered.append(
            f"""      <g class="svg-badge {badge_class}">
        <rect x="{current_x}" y="{y}" width="22" height="14" rx="4"></rect>
        <text x="{current_x + 11}" y="{y + 10}" text-anchor="middle">{badge}</text>
      </g>"""
        )
        current_x += 26
    return "\n".join(rendered)


def _is_self_fk(fk: ForeignKeyInfo) -> bool:
    return fk.child_key == fk.parent_key


def _self_fks_by_table(foreign_keys: Iterable[ForeignKeyInfo]) -> dict[str, list[ForeignKeyInfo]]:
    grouped: dict[str, list[ForeignKeyInfo]] = {}
    for fk in foreign_keys:
        if _is_self_fk(fk):
            grouped.setdefault(fk.child_key, []).append(fk)
    return grouped


def _render_diagram_sections(
    metadata: SchemaMetadata,
    sections: list[DiagramSection],
) -> str:
    if not metadata.tables:
        return '<div class="canvas"><p style="padding:18px;color:#94a3b8">No tables found for selected schemas.</p></div>'

    return "\n".join(
        f"""<article class="diagram-section lineage-{html.escape(section.lineage)}">
  <div class="section-header">
    <h2>{html.escape(section.title)}</h2>
    <p>{html.escape(section.description)}</p>
  </div>
  {_render_section_body(metadata, section)}
</article>"""
        for section in sections
    )


def _render_section_body(metadata: SchemaMetadata, section: DiagramSection) -> str:
    if section.lanes:
        return f"""<div class="workflow-lanes">
{_render_workflow_lanes(metadata, section)}
  </div>"""
    return f"""<div class="canvas">{_render_svg(metadata, section)}</div>"""


def _render_workflow_lanes(metadata: SchemaMetadata, section: DiagramSection) -> str:
    rendered_lanes = []
    for lane in section.lanes:
        lane_section = DiagramSection(
            title=lane.title,
            description=lane.description,
            table_groups=lane.table_groups,
            lineage=lane.lineage,
            drawn_fk_names=lane.drawn_fk_names,
            logical_edges=lane.logical_edges,
            table_lineages=section.table_lineages,
            fk_lineages=section.fk_lineages,
            fk_labels=section.fk_labels,
        )
        rendered_lanes.append(
            f"""    <section class="workflow-lane lineage-{html.escape(lane.lineage)}">
      <div class="workflow-lane-header">
        <h3>{html.escape(lane.title)}</h3>
        <p>{html.escape(lane.description)}</p>
      </div>
      <div class="canvas">{_render_svg(metadata, lane_section)}</div>
    </section>"""
        )
    return "\n".join(rendered_lanes)


def _render_svg(metadata: SchemaMetadata, section: DiagramSection | None = None) -> str:
    if not metadata.tables:
        return '<p style="padding:18px;color:#94a3b8">No tables found for selected schemas.</p>'

    section_metadata = _section_metadata(metadata, section)
    if not section_metadata.tables:
        return '<p style="padding:18px;color:#94a3b8">No tables found for this section.</p>'

    positions = _layout_tables(section_metadata.tables, section)
    width = max(box["x"] + box["width"] for box in positions.values()) + 40
    height = max(box["y"] + box["height"] for box in positions.values()) + 40
    edges = "\n".join(
        _render_edge(fk, positions, section)
        for fk in section_metadata.foreign_keys
        if not _is_self_fk(fk)
    )
    table_by_name = {table.table_name: table for table in section_metadata.tables}
    logical_edges = "\n".join(
        _render_logical_edge(edge, table_by_name, positions)
        for edge in (section.logical_edges if section else ())
    )
    tables = "\n".join(
        _render_svg_table(table, positions[table.key], _table_lineage(section, table))
        for table in section_metadata.tables
    )

    return f"""<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Entity relationship diagram">
  <defs>
    {_render_svg_markers()}
  </defs>
  <g class="edges">{edges}
{logical_edges}</g>
  <g class="tables">{tables}</g>
</svg>"""


def _section_metadata(
    metadata: SchemaMetadata,
    section: DiagramSection | None,
) -> SchemaMetadata:
    if section is None:
        return metadata

    section_names = set(section.table_names)
    tables = [table for table in metadata.tables if table.table_name in section_names]
    visible_keys = {table.key for table in tables}
    drawn_fk_names = set(section.drawn_fk_names)
    foreign_keys = [
        fk
        for fk in metadata.foreign_keys
        if fk.child_key in visible_keys
        and fk.parent_key in visible_keys
        and fk.fk_name not in section.hidden_fk_names
        and (not drawn_fk_names or fk.fk_name in drawn_fk_names)
    ]
    return SchemaMetadata(tables=tables, foreign_keys=foreign_keys)


def _render_edge(
    fk: ForeignKeyInfo,
    positions: dict[str, dict[str, int]],
    section: DiagramSection | None,
) -> str:
    return _render_relationship_path(
        source_key=fk.parent_key,
        target_key=fk.child_key,
        title=fk.fk_name,
        label=_fk_label(section, fk),
        positions=positions,
        lineage=_fk_lineage(section, fk),
    )


def _render_logical_edge(
    edge: DiagramLogicalEdge,
    table_by_name: dict[str, TableInfo],
    positions: dict[str, dict[str, int]],
) -> str:
    source = table_by_name.get(edge.source_table)
    target = table_by_name.get(edge.target_table)
    if source is None or target is None:
        return ""

    return _render_relationship_path(
        source_key=source.key,
        target_key=target.key,
        title=edge.edge_name,
        label=edge.label,
        positions=positions,
        lineage=edge.lineage,
        logical=True,
    )


def _render_relationship_path(
    *,
    source_key: str,
    target_key: str,
    title: str,
    label: str,
    positions: dict[str, dict[str, int]],
    lineage: str,
    logical: bool = False,
) -> str:
    source = positions.get(source_key)
    target = positions.get(target_key)
    if not source or not target:
        return ""

    source_center_x = source["x"] + source["width"] // 2
    target_center_x = target["x"] + target["width"] // 2
    source_center_y = source["y"] + source["height"] // 2
    target_center_y = target["y"] + target["height"] // 2
    endpoint_pad = 16

    if abs(target_center_x - source_center_x) < 40:
        start_x = source_center_x
        end_x = target_center_x
        if target_center_y >= source_center_y:
            start_y = source["y"] + source["height"] + endpoint_pad
            end_y = target["y"] - endpoint_pad
        else:
            start_y = source["y"] - endpoint_pad
            end_y = target["y"] + target["height"] + endpoint_pad
        path = f"M {start_x} {start_y} V {end_y}"
        label_x = start_x + 8
        label_y = (start_y + end_y) // 2 - 5
    else:
        start_y = source["y"] + min(source["height"] - 18, 62)
        end_y = target["y"] + min(target["height"] - 18, 62)
        if target_center_x >= source_center_x:
            start_x = source["x"] + source["width"] + endpoint_pad
            end_x = target["x"] - endpoint_pad
        else:
            start_x = source["x"] - endpoint_pad
            end_x = target["x"] + target["width"] + endpoint_pad

        mid_x = (start_x + end_x) // 2
        path = (
            f"M {start_x} {start_y} H {end_x}"
            if start_y == end_y
            else f"M {start_x} {start_y} H {mid_x} V {end_y} H {end_x}"
        )
        label_x = mid_x + 6
        label_y = (start_y + end_y) // 2 - 5

    edge_classes = f"relationship-edge edge-{html.escape(lineage)}"
    if logical:
        edge_classes = f"{edge_classes} is-logical"
    label_svg = ""
    if label:
        label_svg = (
            f"""\n      <text class="relationship-label label-{html.escape(lineage)}" """
            f"""x="{label_x}" y="{label_y}">{html.escape(label)}</text>"""
        )

    return f"""    <path class="{edge_classes}" d="{path}" marker-end="url(#arrow-{html.escape(lineage)})">
      <title>{html.escape(title)}</title>
    </path>{label_svg}"""


def _table_lineage(section: DiagramSection | None, table: TableInfo) -> str:
    if section is None:
        return "other"
    return dict(section.table_lineages).get(table.table_name, section.lineage)


def _fk_lineage(section: DiagramSection | None, fk: ForeignKeyInfo) -> str:
    if section is None:
        return "other"
    return dict(section.fk_lineages).get(fk.fk_name, section.lineage)


def _fk_label(section: DiagramSection | None, fk: ForeignKeyInfo) -> str:
    if section is None:
        return ""
    return dict(section.fk_labels).get(fk.fk_name, "")


def _render_svg_table(table: TableInfo, box: dict[str, int], lineage: str) -> str:
    columns = _display_columns(table)
    visible_columns = columns[:10]
    hidden_count = max(0, len(columns) - len(visible_columns))
    rows = []
    y = box["y"] + 54
    for column in visible_columns:
        badges = []
        if column.primary_key:
            badges.append("PK")
        if column.foreign_key:
            badges.append("FK")
        nullable = " null" if column.nullable else ""
        badge_svg = _render_svg_badges(box["x"] + 12, y - 12, badges)
        rows.append(
            f"""      <text x="{box['x'] + 64}" y="{y}" fill="#e2e8f0" font-size="11">{html.escape(column.column_name)}</text>
{badge_svg}
      <text x="{box['x'] + box['width'] - 12}" y="{y}" fill="#94a3b8" font-size="10" text-anchor="end">{html.escape(column.data_type + nullable)}</text>"""
        )
        y += 22
    if hidden_count:
        rows.append(
            f"""      <text x="{box['x'] + 12}" y="{y}" fill="#94a3b8" font-size="11">+ {hidden_count} more columns</text>"""
        )

    title = html.escape(table.table_name)
    schema = html.escape(table.schema_name)
    search = html.escape(_table_search_text(table))
    return f"""    <g class="erd-table table-{html.escape(lineage)}" data-search="{search}">
      <rect class="table-shell" x="{box['x']}" y="{box['y']}" width="{box['width']}" height="{box['height']}" rx="8"></rect>
      <rect class="table-head" x="{box['x']}" y="{box['y']}" width="{box['width']}" height="36" rx="8"></rect>
      <text x="{box['x'] + 12}" y="{box['y'] + 23}" fill="#f8fafc" font-size="13" font-weight="700">{title}</text>
      <text x="{box['x'] + box['width'] - 12}" y="{box['y'] + 23}" fill="#38bdf8" font-size="11" text-anchor="end">{schema}</text>
{chr(10).join(rows)}
    </g>"""


def _render_table_cards(metadata: SchemaMetadata) -> str:
    cards = []
    self_fks_by_key = _self_fks_by_table(metadata.foreign_keys)
    for table in metadata.tables:
        column_rows = []
        for column in _display_columns(table):
            badges = []
            if column.primary_key:
                badges.append('<span class="badge pk">PK</span>')
            if column.foreign_key:
                badges.append('<span class="badge fk">FK</span>')
            nullable = " nullable" if column.nullable else ""
            key_badges = f"""<span class="key-badges">{''.join(badges)}</span>"""
            column_rows.append(
                f"""<tr>
  <td class="col-name">{key_badges}{html.escape(column.column_name)}</td>
  <td class="col-type">{html.escape(column.data_type + nullable)}</td>
</tr>"""
            )
        self_fk_notes = "".join(
            f"""<div class="relationship-note">Self FK: {html.escape(fk.child_column)} -> {html.escape(fk.parent_column)}</div>"""
            for fk in self_fks_by_key.get(table.key, [])
        )
        cards.append(
            f"""<article class="table-card" data-search="{html.escape(_table_search_text(table))}">
  <header>
    <h3>{html.escape(table.schema_name)}.{html.escape(table.table_name)}</h3>
    <div class="meta">{table.row_count:,} rows | {len(table.columns)} columns</div>
  </header>
  <table class="columns"><tbody>{''.join(column_rows)}</tbody></table>
  {self_fk_notes}
</article>"""
        )
    return "\n".join(cards)


def _layout_tables(
    tables: list[TableInfo],
    section: DiagramSection | None = None,
) -> dict[str, dict[str, int]]:
    table_width = 320
    x_gap = 120
    y_gap = 70
    positions: dict[str, dict[str, int]] = {}

    if section is not None:
        table_by_name = {table.table_name: table for table in tables}
        for col_index, group in enumerate(section.table_groups):
            y = 24
            for table_name in group:
                table = table_by_name.get(table_name)
                if table is None:
                    continue
                height = _table_box_height(table)
                positions[table.key] = {
                    "x": 24 + col_index * (table_width + x_gap),
                    "y": y,
                    "width": table_width,
                    "height": height,
                }
                y += height + y_gap
        return positions

    columns = max(1, min(4, math.ceil(math.sqrt(len(tables)))))
    row_y = 24
    current_row_height = 0
    for index, table in enumerate(tables):
        grid_col = index % columns
        if grid_col == 0 and index:
            row_y += current_row_height + y_gap
            current_row_height = 0
        height = _table_box_height(table)
        current_row_height = max(current_row_height, height)
        positions[table.key] = {
            "x": 24 + grid_col * (table_width + x_gap),
            "y": row_y,
            "width": table_width,
            "height": height,
        }

    return positions


def _table_box_height(table: TableInfo) -> int:
    return 58 + min(len(table.columns), 10) * 22 + (22 if len(table.columns) > 10 else 0)


def _metadata_json(metadata: SchemaMetadata) -> str:
    return json.dumps(_metadata_dict(metadata), separators=(",", ":")).replace("</", "<\\/")


def _metadata_dict(metadata: SchemaMetadata) -> dict[str, object]:
    return {
        "tables": [
            {
                "schema": table.schema_name,
                "name": table.table_name,
                "rows": table.row_count,
                "columns": [
                    {
                        "name": column.column_name,
                        "type": column.data_type,
                        "nullable": column.nullable,
                        "primaryKey": column.primary_key,
                        "foreignKey": column.foreign_key,
                    }
                    for column in _display_columns(table)
                ],
            }
            for table in metadata.tables
        ],
        "foreignKeys": [
            {
                "name": fk.fk_name,
                "child": f"{fk.child_schema}.{fk.child_table}.{fk.child_column}",
                "parent": f"{fk.parent_schema}.{fk.parent_table}.{fk.parent_column}",
            }
            for fk in metadata.foreign_keys
        ],
    }


def _table_search_text(table: TableInfo) -> str:
    values = [table.schema_name, table.table_name]
    values.extend(column.column_name for column in table.columns)
    values.extend(column.data_type for column in table.columns)
    return " ".join(values).lower()


def _diagram_identifier(schema_name: str, table_name: str) -> str:
    raw = table_name if schema_name else table_name
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if identifier and identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier.upper()


def _mermaid_type(data_type: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", data_type).strip("_")
    return base or "unknown"


def _format_data_type(type_name: str, max_length: int, precision: int, scale: int) -> str:
    lower = type_name.lower()
    if lower in {"nvarchar", "nchar"}:
        length = "max" if max_length == -1 else str(max_length // 2)
        return f"{lower}({length})"
    if lower in {"varchar", "char", "varbinary", "binary"}:
        length = "max" if max_length == -1 else str(max_length)
        return f"{lower}({length})"
    if lower in {"decimal", "numeric"}:
        return f"{lower}({precision},{scale})"
    if lower in {"datetime2", "datetimeoffset", "time"}:
        return f"{lower}({scale})"
    return lower


_COLUMN_SQL = text(
    """
    WITH row_counts AS (
        SELECT
            object_id,
            SUM(CASE WHEN index_id IN (0, 1) THEN row_count ELSE 0 END) AS row_count
        FROM sys.dm_db_partition_stats
        GROUP BY object_id
    ),
    pk_columns AS (
        SELECT
            ic.object_id,
            ic.column_id
        FROM sys.indexes i
        JOIN sys.index_columns ic
          ON ic.object_id = i.object_id
         AND ic.index_id = i.index_id
        WHERE i.is_primary_key = 1
    ),
    fk_columns AS (
        SELECT DISTINCT
            parent_object_id AS object_id,
            parent_column_id AS column_id
        FROM sys.foreign_key_columns
    )
    SELECT
        s.name AS schema_name,
        t.name AS table_name,
        c.column_id,
        c.name AS column_name,
        ty.name AS type_name,
        c.max_length,
        c.precision,
        c.scale,
        c.is_nullable,
        CASE WHEN pk.column_id IS NULL THEN 0 ELSE 1 END AS is_primary_key,
        CASE WHEN fk.column_id IS NULL THEN 0 ELSE 1 END AS is_foreign_key,
        COALESCE(rc.row_count, 0) AS row_count
    FROM sys.tables t
    JOIN sys.schemas s
      ON s.schema_id = t.schema_id
    JOIN sys.columns c
      ON c.object_id = t.object_id
    JOIN sys.types ty
      ON ty.user_type_id = c.user_type_id
    LEFT JOIN row_counts rc
      ON rc.object_id = t.object_id
    LEFT JOIN pk_columns pk
      ON pk.object_id = c.object_id
     AND pk.column_id = c.column_id
    LEFT JOIN fk_columns fk
      ON fk.object_id = c.object_id
     AND fk.column_id = c.column_id
    WHERE t.is_ms_shipped = 0
      AND s.name IN :schema_names
    ORDER BY s.name, t.name, c.column_id
    """
)

_FOREIGN_KEY_SQL = text(
    """
    SELECT
        fk.name AS fk_name,
        child_schema.name AS child_schema,
        child_table.name AS child_table,
        child_col.name AS child_column,
        parent_schema.name AS parent_schema,
        parent_table.name AS parent_table,
        parent_col.name AS parent_column,
        fkc.constraint_column_id
    FROM sys.foreign_keys fk
    JOIN sys.foreign_key_columns fkc
      ON fkc.constraint_object_id = fk.object_id
    JOIN sys.tables child_table
      ON child_table.object_id = fkc.parent_object_id
    JOIN sys.schemas child_schema
      ON child_schema.schema_id = child_table.schema_id
    JOIN sys.columns child_col
      ON child_col.object_id = child_table.object_id
     AND child_col.column_id = fkc.parent_column_id
    JOIN sys.tables parent_table
      ON parent_table.object_id = fkc.referenced_object_id
    JOIN sys.schemas parent_schema
      ON parent_schema.schema_id = parent_table.schema_id
    JOIN sys.columns parent_col
      ON parent_col.object_id = parent_table.object_id
     AND parent_col.column_id = fkc.referenced_column_id
    WHERE child_schema.name IN :schema_names
       OR parent_schema.name IN :schema_names
    ORDER BY child_schema.name, child_table.name, fk.name, fkc.constraint_column_id
    """
)
