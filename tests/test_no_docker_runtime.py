from __future__ import annotations

import json
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from pricing_pipeline.config import Settings
from scripts import no_docker_services

LOCAL_AIRFLOW_ENV_KEYS = [
    "AIRFLOW_HOME",
    "AIRFLOW__CORE__DAGS_FOLDER",
    "AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS",
    "AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE",
    "PRICING_MIGRATIONS_DIR",
    "PRICING_PROJECT_ROOT",
    "RATING_EXPORT_ROOT",
]


def clear_local_airflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LOCAL_AIRFLOW_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_no_docker_env_example_targets_host_processes_and_external_sql():
    env_example = Path(".env.nodocker.example")

    assert env_example.exists()
    text = env_example.read_text(encoding="utf-8")

    assert "MSSQL_SERVER=<server-name>.database.windows.net,1433" in text
    assert "PRICING_SKIP_DATABASE_CREATE=true" in text
    assert "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" in text
    assert "RATING_EXPORT_ROOT=state/rating_exports" in text
    assert "mssql,1433" not in text
    assert "/opt/pricing" not in text


def test_no_docker_scripts_exist_without_compose_dependency():
    for script in [
        Path("scripts/bootstrap_no_docker.sh"),
        Path("scripts/no_docker_services.py"),
        Path("scripts/start_no_docker_runtime.sh"),
        Path("scripts/start_no_docker_stack.sh"),
        Path("scripts/start_airflow_local.py"),
        Path("scripts/start_mlflow_local.py"),
        Path("scripts/run_pipeline_no_airflow.py"),
    ]:
        assert script.exists(), f"{script} is missing"
        text = script.read_text(encoding="utf-8")
        if script.name not in {"no_docker_services.py", "start_no_docker_stack.sh"}:
            assert "docker compose" not in text.lower()


def test_settings_can_skip_database_creation_for_hosted_targets():
    settings = Settings.from_env({"PRICING_SKIP_DATABASE_CREATE": "true"})

    assert settings.skip_database_create is True


def test_dag_migrations_dir_can_be_overridden_for_no_docker(monkeypatch, tmp_path):
    monkeypatch.setenv("PRICING_MIGRATIONS_DIR", str(tmp_path))

    airflow_module = types.ModuleType("airflow")
    airflow_sdk_module = types.ModuleType("airflow.sdk")

    class FakeTaskOutput:
        def __rshift__(self, other):
            return other

    def dag(**dag_kwargs):
        def decorator(func):
            def factory(*args, **kwargs):
                func(*args, **kwargs)
                return types.SimpleNamespace(dag_id=dag_kwargs["dag_id"])

            return factory

        return decorator

    def task(func):
        def task_factory(*args, **kwargs):
            return FakeTaskOutput()

        return task_factory

    airflow_sdk_module.dag = dag
    airflow_sdk_module.get_current_context = lambda: {}
    airflow_sdk_module.task = task
    airflow_module.sdk = airflow_sdk_module
    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.sdk", airflow_sdk_module)

    dag_path = Path("dags/pricing_superglm_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("pricing_superglm_pipeline_override", dag_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.MIGRATIONS_DIR == tmp_path


def test_no_airflow_runner_help_runs_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline_no_airflow.py", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "--ensure-database" in result.stdout
    assert "--skip-raw-load" in result.stdout
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
    assert "airflow" in result.stdout
    assert "migrate" in result.stdout
    assert "cloudbeaver" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_no_docker_service_picker_builds_python_commands():
    commands = no_docker_services.selected_commands(
        ["migrate", "load-raw-replace", "pipeline"],
        python_executable="/python",
    )

    assert [command.name for command in commands] == [
        "migrate",
        "load-raw-replace",
        "pipeline",
    ]
    assert commands[0].argv == ["/python", "scripts/apply_sql_migrations.py"]
    assert commands[1].argv == ["/python", "scripts/load_fremtpl_raw.py", "--replace"]
    assert commands[2].argv == ["/python", "scripts/run_pipeline_no_airflow.py"]
    assert not any("docker" in part.lower() for command in commands for part in command.argv)


def test_runtime_manager_starts_and_stops_long_running_service(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        category="service",
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
        category="service",
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
        name="migrate",
        description="Apply migrations",
        argv=["python", "migrate.py"],
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = kwargs["stdout"]
        stdout.write("migration ok\n")
        return subprocess.CompletedProcess(argv, 0)

    manager = no_docker_services.RuntimeManager(
        [command],
        log_dir=tmp_path,
        run_factory=fake_run,
    )

    manager.toggle("migrate")

    assert manager.status("migrate") == "succeeded"
    assert calls[0][0] == ["python", "migrate.py"]
    assert (tmp_path / "migrate.log").read_text(encoding="utf-8").endswith("migration ok\n")


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
        category="service",
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
    assert "[STOPPED" in text
    assert "Enter/Space start/stop" in text
    assert "line 2" in text


def test_runtime_screen_lines_show_window_dressing_and_log_scroll_help(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        category="service",
        long_running=True,
    )
    manager = no_docker_services.RuntimeManager([command], log_dir=tmp_path)

    lines = no_docker_services.runtime_screen_lines(
        manager,
        cursor_index=0,
        show_logs=True,
    )
    text = "\n".join(lines)

    assert "+-- No-Docker Runtime Manager" in text
    assert "-- Services " in text
    assert "-- Logs: airflow" in text
    assert "PageUp/u scroll up" in text
    assert "PageDown/d scroll down" in text
    assert "End tail" in text


def test_runtime_screen_lines_can_scroll_log_window(tmp_path):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        category="service",
        long_running=True,
    )
    log_file = tmp_path / "airflow.log"
    log_file.write_text(
        "\n".join(f"log line {number:02d}" for number in range(1, 31)) + "\n",
        encoding="utf-8",
    )
    manager = no_docker_services.RuntimeManager([command], log_dir=tmp_path)

    tail_lines = no_docker_services.runtime_screen_lines(
        manager,
        cursor_index=0,
        show_logs=True,
        log_scroll=0,
        max_log_lines=3,
    )
    scrolled_lines = no_docker_services.runtime_screen_lines(
        manager,
        cursor_index=0,
        show_logs=True,
        log_scroll=2,
        max_log_lines=3,
    )

    tail_text = "\n".join(tail_lines)
    scrolled_text = "\n".join(scrolled_lines)

    assert "log line 28" in tail_text
    assert "log line 30" in tail_text
    assert "log line 27" not in tail_text
    assert "log line 26" in scrolled_text
    assert "log line 28" in scrolled_text
    assert "log line 29" not in scrolled_text
    assert "scroll 2 from tail" in scrolled_text


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
    assert text.index("Services") < text.index("airflow")
    assert text.index("Services") < text.index("cloudbeaver")
    assert text.index("Pipeline Tasks") < text.index("migrate")
    assert text.index("Pipeline Tasks") < text.index("seed-demo")
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

    assert lines[row_index].startswith("> migrate")


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


def test_runtime_tui_page_keys_update_log_scroll(monkeypatch):
    command = no_docker_services.ServiceCommand(
        name="airflow",
        description="Start Airflow",
        argv=["python", "airflow.py"],
        category="service",
        long_running=True,
    )
    manager = no_docker_services.RuntimeManager([command])
    observed_scroll: list[int] = []

    class FakeScreen:
        def __init__(self) -> None:
            self.keys = iter(
                [
                    no_docker_services.curses.KEY_PPAGE,
                    no_docker_services.curses.KEY_NPAGE,
                    no_docker_services.curses.KEY_END,
                    ord("q"),
                ]
            )

        def keypad(self, flag):
            return None

        def timeout(self, milliseconds):
            return None

        def getch(self):
            return next(self.keys)

    def fake_wrapper(callback):
        callback(FakeScreen())

    def fake_draw_runtime_screen(stdscr, manager, *, cursor_index, show_logs, log_scroll):
        observed_scroll.append(log_scroll)

    monkeypatch.setattr(no_docker_services.curses, "wrapper", fake_wrapper)
    monkeypatch.setattr(no_docker_services, "_draw_runtime_screen", fake_draw_runtime_screen)
    monkeypatch.setattr(manager, "stop_all", lambda: None)

    no_docker_services.run_runtime_tui(manager=manager)

    assert observed_scroll == [0, no_docker_services.LOG_SCROLL_STEP, 0, 0]


def test_runtime_draw_screen_renders_logs_in_fixed_pane(tmp_path):
    commands = [
        no_docker_services.ServiceCommand(
            name=f"service-{index}",
            description=f"Service {index}",
            argv=["python", "service.py"],
            category="service",
            long_running=True,
        )
        for index in range(12)
    ]
    manager = no_docker_services.RuntimeManager(commands, log_dir=tmp_path)
    (tmp_path / "service-0.log").write_text("visible log line\n", encoding="utf-8")
    screen = FakeCursesScreen(height=12, width=100)

    no_docker_services._draw_runtime_screen(
        screen,
        manager,
        cursor_index=0,
        show_logs=True,
        log_scroll=0,
    )

    rendered_text = "\n".join(text for _, _, text, _ in screen.rendered)

    assert "-- Logs: service-0" in rendered_text
    assert "visible log line" in rendered_text


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
    assert "--services airflow,mlflow" in result.stdout
    assert "--services SERVICES" in result.stdout


def test_interactive_shell_launcher_dry_run_selected_services():
    result = subprocess.run(
        [
            "bash",
            "scripts/start_no_docker_stack.sh",
            "--dry-run",
            "--services",
            "migrate,mlflow,airflow",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "scripts/apply_sql_migrations.py" in result.stdout
    assert "scripts/start_mlflow_local.py" in result.stdout
    assert "scripts/start_airflow_local.py" in result.stdout
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
            "migrate",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "scripts/apply_sql_migrations.py" in result.stdout
    assert "no coprocess" not in result.stderr


def test_start_no_docker_runtime_alias_invokes_launcher():
    result = subprocess.run(
        [
            "bash",
            "scripts/start_no_docker_runtime.sh",
            "--dry-run",
            "--services",
            "migrate",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "scripts/apply_sql_migrations.py" in result.stdout


def test_local_airflow_maps_docker_mount_paths_to_repo_paths(monkeypatch, tmp_path):
    from scripts import start_airflow_local

    clear_local_airflow_env(monkeypatch)

    def fake_load_env():
        monkeypatch.setenv("RATING_EXPORT_ROOT", "/opt/pricing/state/rating_exports")
        monkeypatch.setenv("PRICING_MIGRATIONS_DIR", "/opt/pricing/db/migrations")
        monkeypatch.setenv("PRICING_PROJECT_ROOT", "/opt/pricing")

    captured_exec: dict[str, object] = {}

    def fake_execv(executable: str, command: list[str]) -> None:
        captured_exec["executable"] = executable
        captured_exec["command"] = command
        raise SystemExit(0)

    monkeypatch.setattr(start_airflow_local, "ROOT", tmp_path)
    monkeypatch.setattr(start_airflow_local, "load_env", fake_load_env)
    monkeypatch.setattr(start_airflow_local.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        start_airflow_local,
        "parse_args",
        lambda: types.SimpleNamespace(airflow_args=["version"]),
    )
    monkeypatch.setattr(start_airflow_local.shutil, "which", lambda name: "/usr/bin/airflow")
    monkeypatch.setattr(start_airflow_local.os, "execv", fake_execv)

    with pytest.raises(SystemExit) as exit_info:
        start_airflow_local.main()

    assert exit_info.value.code == 0
    assert Path(os.environ["RATING_EXPORT_ROOT"]) == tmp_path / "state/rating_exports"
    assert Path(os.environ["PRICING_MIGRATIONS_DIR"]) == tmp_path / "db/migrations"
    assert Path(os.environ["PRICING_PROJECT_ROOT"]) == tmp_path
    assert (tmp_path / "state/rating_exports").is_dir()
    assert captured_exec["command"] == ["/usr/bin/airflow", "version"]


def test_local_airflow_configures_predictable_simple_auth(monkeypatch, tmp_path):
    from scripts import start_airflow_local

    clear_local_airflow_env(monkeypatch)

    captured_exec: dict[str, object] = {}

    def fake_execv(executable: str, command: list[str]) -> None:
        captured_exec["executable"] = executable
        captured_exec["command"] = command
        raise SystemExit(0)

    monkeypatch.setattr(start_airflow_local, "ROOT", tmp_path)
    monkeypatch.setattr(start_airflow_local, "load_env", lambda: None)
    monkeypatch.setattr(start_airflow_local.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        start_airflow_local,
        "parse_args",
        lambda: types.SimpleNamespace(airflow_args=["version"]),
    )
    monkeypatch.setattr(start_airflow_local.shutil, "which", lambda name: "/usr/bin/airflow")
    monkeypatch.setattr(start_airflow_local.os, "execv", fake_execv)

    with pytest.raises(SystemExit) as exit_info:
        start_airflow_local.main()

    password_file = tmp_path / "state/airflow/simple_auth_manager_passwords.json"
    assert exit_info.value.code == 0
    assert os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS"] == "admin:admin"
    assert Path(os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"]) == password_file
    assert json.loads(password_file.read_text(encoding="utf-8")) == {"admin": "admin"}
    assert captured_exec["command"] == ["/usr/bin/airflow", "version"]


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
