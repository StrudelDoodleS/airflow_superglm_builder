# Script command index

Script paths stay flat because notebooks, shell launchers, tests, and work
runbooks call them directly. This index groups them by job without breaking
those stable entry points.

## Notebook workspace

| Script | Purpose |
|---|---|
| `scaffold_pricing_model.py` | Create the five-notebook model workspace; reads optional scaffold TOML defaults. |

## SQL schema and inspection

| Script | Purpose |
|---|---|
| `apply_schema.py` | Guard target database and apply only missing versioned migrations. |
| `reset_remote_pricing_schema.py` | Dry-run or destructively reset runtime-owned schemas and replay migrations. |
| `generate_db_diagrams.py` | Generate a catalog-derived static ERD site. |
| `render_schema_diagrams.py` | Render committed Mermaid diagrams and optionally preview them with Chafa Kitty output. |
| `inspect_rating_package.py` | Print one package, its terms, and sample cells. |
| `render_schema_sql.py` | Internal renderer used to substitute configured schema names; not a CLI. |
| `pricing_db.py` | Shared runtime/engine loader used by scripts; not a CLI. |

## Demo data

| Script | Purpose |
|---|---|
| `load_fremtpl_raw.py` | Load the freMTPL demo table; `--replace` truncates/reloads it. |
| `reset_pricing_experiments.py` | Delete local experiment history; requires `--yes`. |

## Local development services

| Script | Purpose |
|---|---|
| `bootstrap_no_docker.sh` | Install dependencies and create local state folders. |
| `no_docker_services.py` | List/start/stop the host-process service menu. |
| `start_no_docker_stack.sh` | Shell launcher for selected host services. |
| `start_mlflow_local.py` | Start optional local MLflow. Notebook publication does not require it. |
| `smoke_check.py` | Check that the installed SuperGLM exposes rating export. |

Use `uv run python <script> --help` before mutations. The destructive commands
are `reset_remote_pricing_schema.py`, `reset_pricing_experiments.py`, and
`load_fremtpl_raw.py --replace`; each has an explicit confirmation or flag.

See the [SQL runbook](../docs/sql/README.md) for the migration/reset decision and
the [notebook guide](../docs/notebooks/README.md) for scaffold usage.
