from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

from pricing_pipeline.infra.config import Settings
from scripts import no_docker_services


def test_no_docker_env_example_targets_host_processes_and_external_sql():
    env_example = Path(".env.nodocker.example")

    assert env_example.exists()
    text = env_example.read_text(encoding="utf-8")

    assert "PRICING_RUNTIME_MODULE=work_runtime.database" in text
    assert "PRICING_PROJECT_ROOT=." in text
    assert "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" in text
    assert "MLFLOW_BACKEND_STORE_URI=sqlite:///state/no_docker/mlflow/mlflow.db" in text
    assert "MLFLOW_ARTIFACT_ROOT=state/no_docker/mlflow/artifacts" in text
    assert "RATING_EXPORT_ROOT=state/no_docker/rating_exports" in text
    assert "MSSQL_SERVER=" not in text
    assert "MSSQL_DATABASE=" not in text
    assert "MSSQL_AUTH_MODE=" not in text
    assert "PRICING_SCHEMA=python_pricing" not in text
    assert "MLOPS_SCHEMA=python_mlops" not in text
    assert "mssql,1433" not in text
    assert "/opt/pricing" not in text


def test_no_docker_scripts_exist_without_compose_dependency():
    assert not Path("scripts/start_no_docker_runtime.sh").exists()
    assert not Path("scripts/apply_sql_migrations.py").exists()

    for script in [
        Path("scripts/apply_schema.py"),
        Path("scripts/bootstrap_no_docker.sh"),
        Path("scripts/no_docker_services.py"),
        Path("scripts/start_no_docker_stack.sh"),
        Path("scripts/start_mlflow_local.py"),
    ]:
        assert script.exists(), f"{script} is missing"
        text = script.read_text(encoding="utf-8")
        if script.name not in {"no_docker_services.py", "start_no_docker_stack.sh"}:
            assert "docker compose" not in text.lower()


def test_settings_can_skip_database_creation_for_hosted_targets():
    settings = Settings.from_env({"PRICING_SKIP_DATABASE_CREATE": "true"})

    assert settings.skip_database_create is True


def test_apply_schema_script_starts_without_pythonpath(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PRICING_SCHEMA_DIR"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "scripts/apply_schema.py"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "No schema DDL files found" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_no_docker_service_picker_lists_available_services_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/no_docker_services.py", "list"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "mlflow" in result.stdout
    assert "apply-schema" in result.stdout
    assert "migrate" not in result.stdout
    assert "cloudbeaver" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_no_docker_service_picker_builds_python_commands():
    commands = no_docker_services.selected_commands(
        ["apply-schema", "load-fremtpl-replace"],
        python_executable="/python",
    )

    assert [command.name for command in commands] == [
        "apply-schema",
        "load-fremtpl-replace",
    ]
    assert commands[0].argv == ["/python", "scripts/apply_schema.py"]
    assert commands[1].argv == ["/python", "scripts/load_fremtpl_raw.py", "--replace"]
    assert not any("docker" in part.lower() for command in commands for part in command.argv)


def test_runtime_manager_starts_and_stops_long_running_service(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        long_running=True,
    )
    created_processes: list[FakeProcess] = []

    def fake_popen(argv, **kwargs):
        process = FakeProcess(argv=argv, kwargs=kwargs)
        created_processes.append(process)
        return process

    manager = no_docker_services.RuntimeManager(
        [command],
        log_dir=tmp_path,
        popen_factory=fake_popen,
    )

    manager.toggle("airflow")

    assert manager.status("airflow") == "running"
    assert created_processes[0].argv == ["python", "airflow.py"]
    assert Path(created_processes[0].kwargs["stdout"].name) == tmp_path / "airflow.log"

    manager.toggle("airflow")

    assert created_processes[0].terminated is True
    assert manager.status("airflow") == "stopped"


def test_runtime_manager_stops_long_running_service_process_group(monkeypatch, tmp_path):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        long_running=True,
    )
    created_processes: list[FakeProcess] = []
    sent_signals: list[tuple[int, int]] = []

    def fake_popen(argv, **kwargs):
        process = FakeProcess(argv=argv, kwargs=kwargs)
        process.pid = 12345
        created_processes.append(process)
        return process

    def fake_killpg(process_group_id: int, signal_number: int) -> None:
        sent_signals.append((process_group_id, signal_number))
        created_processes[0].returncode = 0

    monkeypatch.setattr(
        no_docker_services,
        "os",
        types.SimpleNamespace(getpgid=lambda pid: 54321, killpg=fake_killpg),
        raising=False,
    )
    monkeypatch.setattr(no_docker_services, "signal", signal, raising=False)

    manager = no_docker_services.RuntimeManager(
        [command],
        log_dir=tmp_path,
        popen_factory=fake_popen,
    )

    manager.toggle("airflow")

    assert created_processes[0].kwargs["start_new_session"] is True

    manager.toggle("airflow")

    assert sent_signals == [(54321, signal.SIGTERM)]
    assert created_processes[0].terminated is False
    assert manager.status("airflow") == "stopped"


def test_runtime_manager_runs_one_shot_service_to_log(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="apply-schema",
        description="Apply schema",
        argv=["python", "apply_schema.py"],
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = kwargs["stdout"]
        stdout.write("schema ok\n")
        return subprocess.CompletedProcess(argv, 0)

    manager = no_docker_services.RuntimeManager(
        [command],
        log_dir=tmp_path,
        run_factory=fake_run,
    )

    manager.toggle("apply-schema")

    assert manager.status("apply-schema") == "succeeded"
    assert calls[0][0] == ["python", "apply_schema.py"]
    assert (tmp_path / "apply-schema.log").read_text(encoding="utf-8").endswith("schema ok\n")


def test_runtime_manager_marks_missing_command_as_failed(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="cloudbeaver",
        description="Start CloudBeaver",
        argv=["docker", "compose"],
    )

    def fake_run(argv, **kwargs):
        raise FileNotFoundError("docker")

    manager = no_docker_services.RuntimeManager(
        [command],
        log_dir=tmp_path,
        run_factory=fake_run,
    )

    manager.toggle("cloudbeaver")

    assert manager.status("cloudbeaver") == "failed"
    assert "docker" in (tmp_path / "cloudbeaver.log").read_text(encoding="utf-8")


def test_runtime_manager_screen_lines_show_status_and_logs(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="mlflow",
        description="Start MLflow",
        argv=["python", "mlflow.py"],
        long_running=True,
    )
    log_file = tmp_path / "mlflow.log"
    log_file.write_text("line 1\nline 2\n", encoding="utf-8")
    manager = no_docker_services.RuntimeManager([command], log_dir=tmp_path)

    lines = no_docker_services.runtime_screen_lines(
        manager,
        cursor_index=0,
        show_logs=True,
    )

    text = "\n".join(lines)
    assert "> mlflow" in text
    assert "[stopped" in text
    assert "Enter/Space start/stop" in text
    assert "line 2" in text


def test_runtime_screen_groups_services_tasks_and_utilities():
    manager = no_docker_services.RuntimeManager(
        list(no_docker_services.service_catalog(python_executable="/python").values())
    )

    lines = no_docker_services.runtime_screen_lines(
        manager,
        cursor_index=0,
        show_logs=False,
    )
    text = "\n".join(lines)

    assert "Services" in text
    assert "Pipeline Tasks" in text
    assert "Utilities" in text
    assert text.index("Services") < text.index("mlflow")
    assert text.index("Services") < text.index("cloudbeaver")
    assert text.index("Pipeline Tasks") < text.index("apply-schema")
    assert text.index("Pipeline Tasks") < text.index("load-fremtpl")
    assert text.index("Utilities") < text.index("bootstrap")
    assert text.index("Utilities") < text.index("diagrams")


def test_runtime_screen_selected_row_index_skips_section_headers():
    manager = no_docker_services.RuntimeManager(
        list(no_docker_services.service_catalog(python_executable="/python").values())
    )

    row_index = no_docker_services.selected_runtime_row_index(
        manager,
        cursor_index=3,
    )
    lines = no_docker_services.runtime_screen_lines(
        manager,
        cursor_index=3,
        show_logs=False,
    )

    assert lines[row_index].startswith("> load-fremtpl")


def test_runtime_tui_handles_ctrl_c_without_traceback(monkeypatch):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        long_running=True,
    )
    manager = no_docker_services.RuntimeManager([command])
    stopped = []

    def fake_wrapper(callback):
        raise KeyboardInterrupt

    monkeypatch.setattr(no_docker_services.curses, "wrapper", fake_wrapper)
    monkeypatch.setattr(manager, "stop_all", lambda: stopped.append(True))

    no_docker_services.run_runtime_tui(manager=manager)

    assert stopped == [True]


def test_runtime_tui_mouse_wheel_down_moves_one_selector_row(monkeypatch, tmp_path):
    commands = [
        no_docker_services.ServiceCommand(
            name=f"service-{index}",
            description=f"Service {index}",
            argv=["python", "service.py"],
            category="service",
            long_running=True,
        )
        for index in range(4)
    ]
    manager = no_docker_services.RuntimeManager(commands, log_dir=tmp_path)
    screen = InteractiveFakeCursesScreen(
        height=12,
        width=120,
        keys=[no_docker_services.curses.KEY_MOUSE, ord("q")],
    )
    cursor_indexes = []

    monkeypatch.setattr(no_docker_services.curses, "BUTTON5_PRESSED", 0x200000, raising=False)
    monkeypatch.setattr(no_docker_services.curses, "mousemask", lambda *_: None)
    monkeypatch.setattr(no_docker_services.curses, "mouseinterval", lambda *_: None)
    monkeypatch.setattr(
        no_docker_services.curses,
        "getmouse",
        lambda: (0, 0, 0, 0, no_docker_services.curses.BUTTON5_PRESSED),
    )
    monkeypatch.setattr(
        no_docker_services.curses,
        "wrapper",
        lambda callback: callback(screen),
    )
    monkeypatch.setattr(manager, "stop_all", lambda: None)
    monkeypatch.setattr(
        no_docker_services,
        "_draw_runtime_screen",
        lambda _screen, _manager, *, cursor_index, show_logs: cursor_indexes.append(cursor_index),
    )

    no_docker_services.run_runtime_tui(manager=manager)

    assert cursor_indexes == [0, 1]


def test_runtime_draw_screen_keeps_selected_service_visible_without_skipping(tmp_path):
    commands = [
        no_docker_services.ServiceCommand(
            name=f"service-{index}",
            description=f"Service {index}",
            argv=["python", "service.py"],
            category="service",
            long_running=True,
        )
        for index in range(20)
    ]
    manager = no_docker_services.RuntimeManager(commands, log_dir=tmp_path)
    screen = FakeCursesScreen(height=8, width=120)

    no_docker_services._draw_runtime_screen(
        screen,
        manager,
        cursor_index=12,
        show_logs=False,
    )

    rendered_text = "\n".join(text for _, _, text, _ in screen.rendered)

    assert "> service-12" in rendered_text
    assert "service-11" in rendered_text
    assert "service-13" in rendered_text


class FakeCursesScreen:
    def __init__(self, *, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.rendered: list[tuple[int, int, str, int]] = []

    def erase(self):
        return None

    def getmaxyx(self):
        return (self.height, self.width)

    def addnstr(self, row, column, text, limit, attributes):
        self.rendered.append((row, column, text[:limit], attributes))

    def refresh(self):
        return None


class InteractiveFakeCursesScreen(FakeCursesScreen):
    def __init__(self, *, height: int, width: int, keys: list[int]) -> None:
        super().__init__(height=height, width=width)
        self.keys = keys

    def keypad(self, _enabled):
        return None

    def timeout(self, _milliseconds):
        return None

    def getch(self):
        if self.keys:
            return self.keys.pop(0)
        return ord("q")


class FakeProcess:
    def __init__(self, argv, kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_interactive_shell_launcher_help_documents_keyboard_menu():
    result = subprocess.run(
        ["bash", "scripts/start_no_docker_stack.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "persistent runtime TUI" in result.stdout
    assert "--services mlflow" in result.stdout
    assert "--services SERVICES" in result.stdout


def test_interactive_shell_launcher_dry_run_selected_services():
    result = subprocess.run(
        [
            "bash",
            "scripts/start_no_docker_stack.sh",
            "--dry-run",
            "--services",
            "apply-schema,mlflow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "scripts/apply_schema.py" in result.stdout
    assert "scripts/start_mlflow_local.py" in result.stdout
    assert "docker compose" not in result.stdout


def test_interactive_shell_launcher_runs_when_called_with_zsh():
    if shutil.which("zsh") is None:
        return

    result = subprocess.run(
        [
            "zsh",
            "scripts/start_no_docker_stack.sh",
            "--dry-run",
            "--services",
            "apply-schema",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "scripts/apply_schema.py" in result.stdout
    assert "no coprocess" not in result.stderr


def test_apply_schema_direct_script_import_resolves_repo_package():
    script_path = Path("scripts/apply_schema.py").resolve()
    scripts_dir = script_path.parent
    repo_root = script_path.parents[1]
    code = (
        "import runpy, sys\n"
        f"repo_root = {str(repo_root)!r}\n"
        f"scripts_dir = {str(scripts_dir)!r}\n"
        "sys.path = [scripts_dir] + [path for path in sys.path if path not in {'', repo_root}]\n"
        f"runpy.run_path({str(script_path)!r}, run_name='apply_schema_import_check')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_interactive_shell_launcher_cloudbeaver_is_explicitly_docker_backed():
    result = subprocess.run(
        [
            "bash",
            "scripts/start_no_docker_stack.sh",
            "--dry-run",
            "--services",
            "cloudbeaver",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "cloudbeaver uses Docker Compose in this repo" in result.stdout
    assert "docker compose --profile sql-ui up cloudbeaver" in result.stdout
