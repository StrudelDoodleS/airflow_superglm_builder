from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def test_windows_file_lock_uses_one_byte_lock_and_unlock(monkeypatch, tmp_path):
    from pricing_pipeline.infra import file_lock

    calls: list[tuple[int, int, int]] = []
    fake_msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_UNLCK=2,
        locking=lambda descriptor, mode, length: calls.append((descriptor, mode, length)),
    )
    monkeypatch.setattr(file_lock, "_is_windows", lambda: True)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    lock_path = tmp_path / "publication.lock"
    with file_lock.exclusive_file_lock(lock_path) as handle:
        descriptor = handle.fileno()
        assert lock_path.read_bytes() == b"\0"
        assert calls == [(descriptor, fake_msvcrt.LK_LOCK, 1)]

    assert calls == [
        (descriptor, fake_msvcrt.LK_LOCK, 1),
        (descriptor, fake_msvcrt.LK_UNLCK, 1),
    ]


def test_file_lock_consumers_do_not_import_fcntl_at_module_load():
    repository_root = Path(__file__).resolve().parents[1]
    consumer_paths = (
        repository_root / "pricing_pipeline/infra/offline_sqlite.py",
        repository_root / "pricing_pipeline/publishing/editor_candidate.py",
        repository_root / "pricing_pipeline/workbench/submission.py",
    )

    for path in consumer_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top_level_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "fcntl" not in top_level_imports, path
