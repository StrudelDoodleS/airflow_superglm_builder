from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass
import os
import signal
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
PROCESS_STOP_TIMEOUT_SECONDS = 15
SCREEN_WIDTH = 96
LOG_SCROLL_STEP = 10


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


def _process_group_id(process: subprocess.Popen) -> int | None:
    pid = getattr(process, "pid", None)
    if pid is None or not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(pid)
    except (OSError, ProcessLookupError):
        return None


def _signal_process(process: subprocess.Popen, signal_number: int) -> None:
    process_group_id = _process_group_id(process)
    if process_group_id is not None and hasattr(os, "killpg"):
        try:
            os.killpg(process_group_id, signal_number)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass

    if signal_number == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    _signal_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process(process, signal.SIGKILL)
        process.wait()


def _headline(title: str, *, width: int = SCREEN_WIDTH) -> str:
    return f"+-- {title} ".ljust(width - 1, "-") + "+"


def _section(title: str, *, width: int = SCREEN_WIDTH) -> str:
    return f"-- {title} ".ljust(width, "-")


def _log_window(
    lines: list[str],
    *,
    max_lines: int,
    scroll_offset: int,
) -> tuple[list[str], int]:
    if not lines:
        return ["no logs yet"], 0

    max_lines = max(1, max_lines)
    max_scroll = max(0, len(lines) - max_lines)
    normalized_scroll = min(max(scroll_offset, 0), max_scroll)
    end_index = len(lines) - normalized_scroll
    start_index = max(0, end_index - max_lines)
    return lines[start_index:end_index], normalized_scroll


def _scrollbar_thumb_rows(
    *,
    start_row: int,
    height: int,
    total_items: int,
    visible_items: int,
    offset: int,
) -> set[int]:
    if height <= 0 or total_items <= visible_items or total_items <= 0:
        return set()

    visible_items = max(1, min(visible_items, total_items))
    offset = min(max(offset, 0), total_items - visible_items)
    thumb_height = max(1, round(height * (visible_items / total_items)))
    max_thumb_top = max(0, height - thumb_height)
    max_offset = max(1, total_items - visible_items)
    thumb_top = round(max_thumb_top * (offset / max_offset))
    return set(range(start_row + thumb_top, start_row + thumb_top + thumb_height))


def _draw_scrollbar(
    stdscr,
    *,
    column: int,
    start_row: int,
    height: int,
    total_items: int,
    visible_items: int,
    offset: int,
) -> None:
    thumb_rows = _scrollbar_thumb_rows(
        start_row=start_row,
        height=height,
        total_items=total_items,
        visible_items=visible_items,
        offset=offset,
    )
    if not thumb_rows:
        return

    for row in range(start_row, start_row + height):
        character = "#" if row in thumb_rows else "|"
        stdscr.addnstr(row, column, character, 1, curses.A_NORMAL)


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
                    start_new_session=True,
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
        _terminate_process(process)
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
    log_scroll: int = 0,
    max_log_lines: int = 18,
) -> list[str]:
    manager.poll()
    lines = [
        _headline("No-Docker Runtime Manager"),
        "Enter/Space start/stop/run | r restart | l logs | x stop all | q quit",
        "PageUp/u scroll up | PageDown/d scroll down | End tail",
        "",
    ]
    selected_name = manager.names()[cursor_index] if manager.names() else None
    for category, label in RUNTIME_CATEGORIES:
        category_names = [
            name for name in manager.names() if manager.records[name].command.category == category
        ]
        if not category_names:
            continue
        lines.append(_section(label))
        for name in category_names:
            record = manager.records[name]
            cursor = ">" if name == selected_name else " "
            kind = _command_kind(record.command)
            lines.append(
                f"{cursor} {name:<16} [{record.status.upper():<9}] "
                f"{kind:<7} {record.command.description}"
            )
        lines.append("")
    if lines[-1] == "":
        lines.pop()

    if show_logs and manager.names():
        selected = selected_name or manager.names()[0]
        lines.extend([""])
        lines.extend(
            runtime_log_pane_lines(
                manager,
                selected_name=selected,
                log_scroll=log_scroll,
                max_log_lines=max_log_lines,
            )
        )
    return lines


def runtime_log_pane_lines(
    manager: RuntimeManager,
    *,
    selected_name: str,
    log_scroll: int = 0,
    max_log_lines: int = 18,
) -> list[str]:
    log_lines, normalized_scroll = _log_window(
        manager.tail_log(selected_name, max_lines=100_000),
        max_lines=max_log_lines,
        scroll_offset=log_scroll,
    )
    scroll_state = (
        "tail" if normalized_scroll == 0 else f"scroll {normalized_scroll} from tail"
    )
    return [
        _section(f"Logs: {selected_name} ({scroll_state})"),
        str(manager.log_path(selected_name)),
        *log_lines,
    ]


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
    log_scroll: int,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    selected_name = manager.names()[cursor_index]
    service_screen_lines = runtime_screen_lines(
        manager,
        cursor_index=cursor_index,
        show_logs=False,
    )
    selected_row = selected_runtime_row_index(manager, cursor_index=cursor_index)

    if show_logs:
        log_pane_height = min(max(5, height // 3), max(0, height - 5))
        service_end_row = max(4, height - log_pane_height - 1)
    else:
        log_pane_height = 0
        service_end_row = height - 1

    header_lines = service_screen_lines[:3]
    service_lines = service_screen_lines[3:]
    service_start_row = len(header_lines)
    service_height = max(0, service_end_row - service_start_row)
    selected_service_index = max(0, selected_row - len(header_lines))
    service_scroll = max(0, selected_service_index - service_height // 2)
    max_service_scroll = max(0, len(service_lines) - service_height)
    service_scroll = min(service_scroll, max_service_scroll)
    visible_service_lines = service_lines[service_scroll : service_scroll + service_height]

    display_lines = [*header_lines, *visible_service_lines]
    for row, line in enumerate(display_lines):
        if row >= height - 1:
            break
        attributes = (
            curses.A_REVERSE
            if line.startswith(f"> {selected_name:<16}")
            else curses.A_NORMAL
        )
        stdscr.addnstr(row, 0, line, max(width - 1, 0), attributes)
    _draw_scrollbar(
        stdscr,
        column=max(width - 1, 0),
        start_row=service_start_row,
        height=service_height,
        total_items=len(service_lines),
        visible_items=len(visible_service_lines),
        offset=service_scroll,
    )

    if show_logs and log_pane_height > 0:
        log_start_row = max(0, height - log_pane_height)
        max_log_lines = max(1, log_pane_height - 3)
        all_log_lines = manager.tail_log(selected_name, max_lines=100_000)
        visible_log_lines, normalized_log_scroll = _log_window(
            all_log_lines,
            max_lines=max_log_lines,
            scroll_offset=log_scroll,
        )
        log_scroll_state = (
            "tail"
            if normalized_log_scroll == 0
            else f"scroll {normalized_log_scroll} from tail"
        )
        log_window_start = max(
            0,
            len(all_log_lines) - normalized_log_scroll - len(visible_log_lines),
        )
        log_lines = [
            _section(f"Logs: {selected_name} ({log_scroll_state})"),
            str(manager.log_path(selected_name)),
            *visible_log_lines,
        ]
        for offset, line in enumerate(log_lines[:log_pane_height]):
            row = log_start_row + offset
            if row >= height - 1:
                break
            stdscr.addnstr(row, 0, line, max(width - 1, 0), curses.A_NORMAL)
        _draw_scrollbar(
            stdscr,
            column=max(width - 1, 0),
            start_row=log_start_row + 2,
            height=max(0, min(max_log_lines, height - log_start_row - 3)),
            total_items=len(all_log_lines),
            visible_items=len(visible_log_lines),
            offset=log_window_start,
        )
    stdscr.refresh()


def run_runtime_tui(manager: RuntimeManager | None = None) -> None:
    manager = manager or RuntimeManager(list(service_catalog().values()))

    def _runtime(stdscr) -> None:
        cursor_index = 0
        show_logs = True
        log_scroll = 0
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
                log_scroll=log_scroll,
            )
            key = stdscr.getch()
            if key == -1:
                continue
            if key in (ord("q"), ord("Q"), 27):
                manager.stop_all()
                return
            if key in (curses.KEY_DOWN, ord("j"), ord("J")):
                cursor_index = (cursor_index + 1) % len(names)
                log_scroll = 0
                continue
            if key in (curses.KEY_UP, ord("k"), ord("K")):
                cursor_index = (cursor_index - 1) % len(names)
                log_scroll = 0
                continue
            if key in (curses.KEY_PPAGE, ord("u"), ord("U")):
                log_scroll += LOG_SCROLL_STEP
                continue
            if key in (curses.KEY_NPAGE, ord("d"), ord("D")):
                log_scroll = max(0, log_scroll - LOG_SCROLL_STEP)
                continue
            if key in (curses.KEY_END, ord("e"), ord("E")):
                log_scroll = 0
                continue
            if key in (ord("l"), ord("L")):
                show_logs = not show_logs
                continue
            if key in (ord("x"), ord("X")):
                manager.stop_all()
                continue
            if key in (ord("r"), ord("R")):
                manager.restart(names[cursor_index])
                log_scroll = 0
                continue
            if key in (ord(" "), 10, 13, curses.KEY_ENTER):
                manager.toggle(names[cursor_index])
                log_scroll = 0

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
                processes.append(
                    subprocess.Popen(command.argv, cwd=ROOT, start_new_session=True)
                )
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
            _terminate_process(process)


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
