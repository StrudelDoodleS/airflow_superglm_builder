from __future__ import annotations

import argparse
import curses
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


DEFAULT_SELECTED = frozenset({"airflow"})


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


def service_names() -> list[str]:
    return list(service_catalog())


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


def parse_services_csv(services_csv: str) -> list[str]:
    names = [name.strip() for name in services_csv.split(",") if name.strip()]
    selected_commands(names)
    return names


def list_services() -> None:
    for command in service_catalog().values():
        kind = "long-running" if command.long_running else "one-shot"
        print(f"{command.name:<16} {kind:<13} {command.description}")


def menu_lines(
    selected: set[str],
    *,
    cursor_index: int,
) -> list[str]:
    names = service_names()
    lines = [
        "No-Docker local launcher",
        "Space toggles, Enter runs, q quits.",
        "",
    ]
    catalog = service_catalog()
    for index, name in enumerate(names):
        cursor = ">" if index == cursor_index else " "
        marker = "x" if name in selected else " "
        command = catalog[name]
        kind = "long-running" if command.long_running else "one-shot"
        lines.append(
            f"{cursor} [{marker}] {index + 1}. {name:<16} {kind:<13} {command.description}"
        )
    lines.extend(
        [
            "",
            "CloudBeaver is Docker Compose backed in this repo; leave it off on Docker-blocked work machines.",
        ]
    )
    return lines


def _draw_menu(stdscr, selected: set[str], cursor_index: int) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    for row, line in enumerate(menu_lines(selected, cursor_index=cursor_index)):
        if row >= height - 1:
            break
        attributes = curses.A_REVERSE if row == cursor_index + 3 else curses.A_NORMAL
        stdscr.addnstr(row, 0, line, max(width - 1, 0), attributes)
    stdscr.refresh()


def choose_services_tui(initial_selected: set[str] | None = None) -> list[str]:
    names = service_names()
    selected = set(initial_selected or DEFAULT_SELECTED)

    def _menu(stdscr) -> list[str]:
        cursor_index = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)

        while True:
            _draw_menu(stdscr, selected, cursor_index)
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                return []
            if key in (ord("r"), ord("R"), 10, 13, curses.KEY_ENTER):
                return [name for name in names if name in selected]
            if key in (curses.KEY_DOWN, ord("j"), ord("J")):
                cursor_index = (cursor_index + 1) % len(names)
                continue
            if key in (curses.KEY_UP, ord("k"), ord("K")):
                cursor_index = (cursor_index - 1) % len(names)
                continue
            if key == ord(" "):
                name = names[cursor_index]
                if name in selected:
                    selected.remove(name)
                else:
                    selected.add(name)
                continue
            if ord("1") <= key <= ord("9"):
                index = key - ord("1")
                if index < len(names):
                    name = names[index]
                    if name in selected:
                        selected.remove(name)
                    else:
                        selected.add(name)

    return curses.wrapper(_menu)


def _run_one_shot(commands: list[ServiceCommand], *, dry_run: bool) -> None:
    for command in commands:
        if command.name == "cloudbeaver":
            print(
                "cloudbeaver uses Docker Compose in this repo; "
                "skip it on Docker-blocked machines.",
                flush=True,
            )
        print(f"==> {command.name}: {' '.join(command.argv)}", flush=True)
        if not dry_run:
            subprocess.run(command.argv, cwd=ROOT, check=True)


def _run_long_running(commands: list[ServiceCommand], *, dry_run: bool) -> None:
    processes: list[subprocess.Popen] = []
    try:
        for command in commands:
            print(f"==> {command.name}: {' '.join(command.argv)}", flush=True)
            if not dry_run:
                processes.append(subprocess.Popen(command.argv, cwd=ROOT))
        if dry_run:
            return
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


def run_services(names: list[str], *, dry_run: bool = False) -> None:
    commands = selected_commands(names)
    one_shot = [command for command in commands if not command.long_running]
    long_running = [command for command in commands if command.long_running]
    _run_one_shot(one_shot, dry_run=dry_run)
    _run_long_running(long_running, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick and run host-process services for the no-Docker workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Show available local services and tasks.")
    menu_parser = subparsers.add_parser("menu", help="Open the interactive TUI menu.")
    menu_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run selected local services/tasks.")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "services",
        nargs="+",
        choices=sorted(service_catalog()),
        help="Services/tasks to run.",
    )
    launcher_parser = subparsers.add_parser(
        "launcher",
        help="Shell-wrapper entrypoint with optional --services CSV.",
        description="Open the TUI menu when --services is omitted.",
        epilog="Example: scripts/start_no_docker_stack.sh --services airflow,mlflow",
    )
    launcher_parser.add_argument("--dry-run", action="store_true")
    launcher_parser.add_argument(
        "--services",
        default=None,
        help="Comma-separated services/tasks. Opens the TUI when omitted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        list_services()
        return
    if args.command == "menu":
        services = choose_services_tui()
        if services:
            run_services(services, dry_run=args.dry_run)
        return
    if args.command == "run":
        run_services(args.services, dry_run=args.dry_run)
        return
    if args.command == "launcher":
        if args.services:
            run_services(parse_services_csv(args.services), dry_run=args.dry_run)
            return
        services = choose_services_tui()
        if services:
            run_services(services, dry_run=args.dry_run)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
