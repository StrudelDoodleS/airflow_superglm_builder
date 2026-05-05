from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServiceCommand:
    name: str
    description: str
    argv: list[str]
    long_running: bool = False


def service_catalog(*, python_executable: str = sys.executable) -> dict[str, ServiceCommand]:
    return {
        "bootstrap": ServiceCommand(
            name="bootstrap",
            description="Create local state folders and install Python dependencies.",
            argv=["bash", "scripts/bootstrap_no_docker.sh"],
        ),
        "mlflow": ServiceCommand(
            name="mlflow",
            description="Start the local MLflow tracking server.",
            argv=[python_executable, "scripts/start_mlflow_local.py"],
            long_running=True,
        ),
        "airflow": ServiceCommand(
            name="airflow",
            description="Start local Airflow standalone.",
            argv=[python_executable, "scripts/start_airflow_local.py"],
            long_running=True,
        ),
        "migrate": ServiceCommand(
            name="migrate",
            description="Apply SQL migrations to the configured pricing database.",
            argv=[python_executable, "scripts/apply_sql_migrations.py"],
        ),
        "load-raw": ServiceCommand(
            name="load-raw",
            description="Load freMTPL raw data if the table is empty.",
            argv=[python_executable, "scripts/load_fremtpl_raw.py"],
        ),
        "load-raw-replace": ServiceCommand(
            name="load-raw-replace",
            description="Truncate and reload freMTPL raw data.",
            argv=[python_executable, "scripts/load_fremtpl_raw.py", "--replace"],
        ),
        "pipeline": ServiceCommand(
            name="pipeline",
            description="Run the full pricing pipeline directly, without Airflow.",
            argv=[python_executable, "scripts/run_pipeline_no_airflow.py"],
        ),
        "diagrams": ServiceCommand(
            name="diagrams",
            description="Generate the local ERD site into state/db_diagrams.",
            argv=[
                python_executable,
                "scripts/generate_db_diagrams.py",
                "--schemas",
                "pricing",
                "--output-dir",
                "state/db_diagrams",
            ],
        ),
        "seed-demo": ServiceCommand(
            name="seed-demo",
            description="Seed simulated pricing model/package history.",
            argv=[python_executable, "scripts/seed_demo_model_variants.py"],
        ),
        "cloudbeaver": ServiceCommand(
            name="cloudbeaver",
            description="Start CloudBeaver SQL UI; Docker-backed local-only option.",
            argv=["docker", "compose", "--profile", "sql-ui", "up", "-d", "cloudbeaver"],
        ),
    }


def selected_commands(
    names: list[str],
    *,
    python_executable: str = sys.executable,
) -> list[ServiceCommand]:
    catalog = service_catalog(python_executable=python_executable)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        choices = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown service(s): {', '.join(unknown)}. Choices: {choices}")
    return [catalog[name] for name in names]


def list_services() -> None:
    for command in service_catalog().values():
        kind = "long-running" if command.long_running else "one-shot"
        print(f"{command.name:<16} {kind:<13} {command.description}")


def _run_one_shot(commands: list[ServiceCommand]) -> None:
    for command in commands:
        print(f"==> {command.name}: {' '.join(command.argv)}", flush=True)
        subprocess.run(command.argv, cwd=ROOT, check=True)


def _run_long_running(commands: list[ServiceCommand]) -> None:
    processes: list[subprocess.Popen] = []
    try:
        for command in commands:
            print(f"==> {command.name}: {' '.join(command.argv)}", flush=True)
            processes.append(subprocess.Popen(command.argv, cwd=ROOT))
        while processes:
            for process in list(processes):
                return_code = process.poll()
                if return_code is not None:
                    processes.remove(process)
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, process.args)
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping selected local services", flush=True)
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_services(names: list[str]) -> None:
    commands = selected_commands(names)
    one_shot = [command for command in commands if not command.long_running]
    long_running = [command for command in commands if command.long_running]
    _run_one_shot(one_shot)
    _run_long_running(long_running)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick and run host-process services for the no-Docker workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Show available local services and tasks.")
    run_parser = subparsers.add_parser("run", help="Run selected local services/tasks.")
    run_parser.add_argument(
        "services",
        nargs="+",
        choices=sorted(service_catalog()),
        help="Services/tasks to run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        list_services()
        return
    if args.command == "run":
        run_services(args.services)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
