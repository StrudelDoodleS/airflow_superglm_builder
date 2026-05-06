from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServiceCommand:
    name: str
    description: str
    argv: list[str]
    category: str = "pipeline-task"
    long_running: bool = False


@dataclass
class RuntimeRecord:
    command: ServiceCommand
    process: Any | None = None
    status: str = "stopped"
    return_code: int | None = None


RUNTIME_LOG_DIR = ROOT / "state/runtime/logs"
RUNTIME_CATEGORIES = (
    ("service", "Services"),
    ("pipeline-task", "Pipeline Tasks"),
    ("utility", "Utilities"),
)


def service_catalog(*, python_executable: str = sys.executable) -> dict[str, ServiceCommand]:
    return {
        "airflow": ServiceCommand(
            name="airflow",
            description="Start local Airflow standalone.",
            argv=[python_executable, "scripts/start_airflow_local.py"],
            category="service",
            long_running=True,
        ),
        "mlflow": ServiceCommand(
            name="mlflow",
            description="Start the local MLflow tracking server.",
            argv=[python_executable, "scripts/start_mlflow_local.py"],
            category="service",
            long_running=True,
        ),
        "cloudbeaver": ServiceCommand(
            name="cloudbeaver",
            description="Start CloudBeaver SQL UI; Docker-backed local-only option.",
            argv=["docker", "compose", "--profile", "sql-ui", "up", "cloudbeaver"],
            category="service",
            long_running=True,
        ),
        "migrate": ServiceCommand(
            name="migrate",
            description="Apply SQL migrations to the configured pricing database.",
            argv=[python_executable, "scripts/apply_sql_migrations.py"],
            category="pipeline-task",
        ),
        "load-raw": ServiceCommand(
            name="load-raw",
            description="Load freMTPL raw data if the table is empty.",
            argv=[python_executable, "scripts/load_fremtpl_raw.py"],
            category="pipeline-task",
        ),
        "load-raw-replace": ServiceCommand(
            name="load-raw-replace",
            description="Truncate and reload freMTPL raw data.",
            argv=[python_executable, "scripts/load_fremtpl_raw.py", "--replace"],
            category="pipeline-task",
        ),
        "pipeline": ServiceCommand(
            name="pipeline",
            description="Run the full pricing pipeline directly, without Airflow.",
            argv=[python_executable, "scripts/run_pipeline_no_airflow.py"],
            category="pipeline-task",
        ),
        "seed-demo": ServiceCommand(
            name="seed-demo",
            description="Seed simulated pricing model/package history.",
            argv=[python_executable, "scripts/seed_demo_model_variants.py"],
            category="pipeline-task",
        ),
        "bootstrap": ServiceCommand(
            name="bootstrap",
            description="Create local state folders and install Python dependencies.",
            argv=["bash", "scripts/bootstrap_no_docker.sh"],
            category="utility",
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
            category="utility",
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


def parse_services_csv(services_csv: str) -> list[str]:
    names = [name.strip() for name in services_csv.split(",") if name.strip()]
    selected_commands(names)
    return names


def list_services() -> None:
    catalog = service_catalog()
    for category, label in RUNTIME_CATEGORIES:
        commands = [command for command in catalog.values() if command.category == category]
        if not commands:
            continue
        print(label)
        for command in commands:
            kind = _command_kind(command)
            print(f"  {command.name:<16} {kind:<13} {command.description}")


class RuntimeManager:
    def __init__(
        self,
        commands: list[ServiceCommand],
        *,
        log_dir: Path = RUNTIME_LOG_DIR,
        popen_factory=subprocess.Popen,
        run_factory=subprocess.run,
    ) -> None:
        self.commands = commands
        self.records = {command.name: RuntimeRecord(command=command) for command in commands}
        self.log_dir = log_dir
        self.popen_factory = popen_factory
        self.run_factory = run_factory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def names(self) -> list[str]:
        return [command.name for command in self.commands]

    def log_path(self, name: str) -> Path:
        return self.log_dir / f"{name}.log"

    def status(self, name: str) -> str:
        self.poll()
        return self.records[name].status

    def toggle(self, name: str) -> None:
        record = self.records[name]
        if record.command.long_running:
            if self.status(name) == "running":
                self.stop(name)
            else:
                self.start(name)
            return
        self.run_one_shot(name)

    def restart(self, name: str) -> None:
        record = self.records[name]
        if not record.command.long_running:
            self.run_one_shot(name)
            return
        if self.status(name) == "running":
            self.stop(name)
        self.start(name)

    def start(self, name: str) -> None:
        record = self.records[name]
        if self.status(name) == "running":
            return
        log_path = self.log_path(name)
        with log_path.open("a", encoding="utf-8") as log_file:
            self._write_header(log_file, record.command)
            try:
                process = self.popen_factory(
                    record.command.argv,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                log_file.write(f"failed to start: {exc}\n")
                record.process = None
                record.status = "failed"
                record.return_code = None
                return
        record.process = process
        record.status = "running"
        record.return_code = None

    def stop(self, name: str) -> None:
        record = self.records[name]
        process = record.process
        if process is None:
            record.status = "stopped"
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record.process = None
        record.status = "stopped"
        record.return_code = process.poll()

    def stop_all(self) -> None:
        for name in self.names():
            if self.records[name].command.long_running:
                self.stop(name)

    def run_one_shot(self, name: str) -> None:
        record = self.records[name]
        log_path = self.log_path(name)
        record.status = "running"
        with log_path.open("a", encoding="utf-8") as log_file:
            self._write_header(log_file, record.command)
            try:
                completed = self.run_factory(
                    record.command.argv,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                log_file.write(f"failed to run: {exc}\n")
                record.return_code = None
                record.status = "failed"
                return
        record.return_code = completed.returncode
        record.status = "succeeded" if completed.returncode == 0 else "failed"

    def poll(self) -> None:
        for record in self.records.values():
            process = record.process
            if process is None:
                continue
            return_code = process.poll()
            if return_code is None:
                record.status = "running"
                continue
            record.process = None
            record.return_code = return_code
            record.status = "succeeded" if return_code == 0 else "failed"

    def tail_log(self, name: str, *, max_lines: int = 18) -> list[str]:
        path = self.log_path(name)
        if not path.exists():
            return ["no logs yet"]
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]

    @staticmethod
    def _write_header(log_file, command: ServiceCommand) -> None:
        log_file.write("\n")
        log_file.write(f"==> {command.name}: {' '.join(command.argv)}\n")
        log_file.flush()


def runtime_screen_lines(
    manager: RuntimeManager,
    *,
    cursor_index: int,
    show_logs: bool,
) -> list[str]:
    manager.poll()
    lines = [
        "No-Docker runtime manager",
        "Enter/Space start/stop/run | r restart | l logs | x stop all | q quit",
        "",
    ]
    selected_name = manager.names()[cursor_index] if manager.names() else None
    for category, label in RUNTIME_CATEGORIES:
        category_names = [
            name for name in manager.names() if manager.records[name].command.category == category
        ]
        if not category_names:
            continue
        lines.append(label)
        for name in category_names:
            record = manager.records[name]
            cursor = ">" if name == selected_name else " "
            kind = _command_kind(record.command)
            lines.append(
                f"{cursor} {name:<16} [{record.status:<9}] {kind:<7} {record.command.description}"
            )
        lines.append("")
    if lines[-1] == "":
        lines.pop()

    if show_logs and manager.names():
        selected = selected_name or manager.names()[0]
        lines.extend(["", f"Logs: {manager.log_path(selected)}", "-" * 72])
        lines.extend(manager.tail_log(selected))
    return lines


def selected_runtime_row_index(manager: RuntimeManager, *, cursor_index: int) -> int:
    selected_name = manager.names()[cursor_index]
    for index, line in enumerate(
        runtime_screen_lines(manager, cursor_index=cursor_index, show_logs=False)
    ):
        if line.startswith(f"> {selected_name:<16}"):
            return index
    return 0


def _command_kind(command: ServiceCommand) -> str:
    if command.category == "service":
        return "service"
    if command.category == "utility":
        return "utility"
    return "task"


def _draw_runtime_screen(
    stdscr,
    manager: RuntimeManager,
    *,
    cursor_index: int,
    show_logs: bool,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    selected_row = selected_runtime_row_index(manager, cursor_index=cursor_index)
    for row, line in enumerate(
        runtime_screen_lines(manager, cursor_index=cursor_index, show_logs=show_logs)
    ):
        if row >= height - 1:
            break
        attributes = curses.A_REVERSE if row == selected_row else curses.A_NORMAL
        stdscr.addnstr(row, 0, line, max(width - 1, 0), attributes)
    stdscr.refresh()


def run_runtime_tui(manager: RuntimeManager | None = None) -> None:
    manager = manager or RuntimeManager(list(service_catalog().values()))

    def _runtime(stdscr) -> None:
        cursor_index = 0
        show_logs = True
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.timeout(500)

        while True:
            names = manager.names()
            _draw_runtime_screen(
                stdscr,
                manager,
                cursor_index=cursor_index,
                show_logs=show_logs,
            )
            key = stdscr.getch()
            if key == -1:
                continue
            if key in (ord("q"), ord("Q"), 27):
                manager.stop_all()
                return
            if key in (curses.KEY_DOWN, ord("j"), ord("J")):
                cursor_index = (cursor_index + 1) % len(names)
                continue
            if key in (curses.KEY_UP, ord("k"), ord("K")):
                cursor_index = (cursor_index - 1) % len(names)
                continue
            if key in (ord("l"), ord("L")):
                show_logs = not show_logs
                continue
            if key in (ord("x"), ord("X")):
                manager.stop_all()
                continue
            if key in (ord("r"), ord("R")):
                manager.restart(names[cursor_index])
                continue
            if key in (ord(" "), 10, 13, curses.KEY_ENTER):
                manager.toggle(names[cursor_index])

    try:
        curses.wrapper(_runtime)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()


def _run_one_shot(commands: list[ServiceCommand], *, dry_run: bool) -> None:
    for command in commands:
        _print_command_warning(command)
        print(f"==> {command.name}: {' '.join(command.argv)}", flush=True)
        if not dry_run:
            subprocess.run(command.argv, cwd=ROOT, check=True)


def _run_long_running(commands: list[ServiceCommand], *, dry_run: bool) -> None:
    processes: list[subprocess.Popen] = []
    try:
        for command in commands:
            _print_command_warning(command)
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


def _print_command_warning(command: ServiceCommand) -> None:
    if command.name == "cloudbeaver":
        print(
            "cloudbeaver uses Docker Compose in this repo; "
            "skip it on Docker-blocked machines.",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick and run host-process services for the no-Docker workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Show available local services and tasks.")
    menu_parser = subparsers.add_parser("menu", help="Open the persistent runtime TUI.")
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
        description="Open the persistent runtime TUI when --services is omitted.",
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
        if args.dry_run:
            print("dry-run runtime TUI: no processes started")
            return
        run_runtime_tui()
        return
    if args.command == "run":
        run_services(args.services, dry_run=args.dry_run)
        return
    if args.command == "launcher":
        if args.services:
            run_services(parse_services_csv(args.services), dry_run=args.dry_run)
            return
        if args.dry_run:
            print("dry-run runtime TUI: no processes started")
            return
        run_runtime_tui()
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
