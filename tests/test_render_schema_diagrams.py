from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.render_schema_diagrams import render_schema_diagrams


def test_standalone_mermaid_sources_match_the_sql_readme_blocks():
    guide = Path("docs/sql/README.md").read_text(encoding="utf-8")
    embedded = re.findall(r"```mermaid\n(.*?)\n```", guide, flags=re.DOTALL)
    standalone = [
        path.read_text(encoding="utf-8").strip()
        for path in sorted(Path("docs/sql/diagrams").glob("*.mmd"))
    ]

    assert standalone == embedded


def test_mermaid_config_uses_svg_native_labels_for_chafa():
    config = Path("docs/sql/diagrams/mermaid.config.json").read_text(encoding="utf-8")

    assert '"htmlLabels": false' in config


def test_render_and_preview_uses_mermaid_then_chafa_kitty(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "01_example.mmd"
    source.write_text("flowchart LR\n    A --> B\n", encoding="utf-8")
    output_dir = tmp_path / "rendered"
    config_file = tmp_path / "mermaid.config.json"
    config_file.write_text('{"htmlLabels": false}\n', encoding="utf-8")
    calls: list[tuple[list[str], bool]] = []

    def fake_runner(command: list[str], *, check: bool) -> None:
        calls.append((command, check))

    rendered = render_schema_diagrams(
        source_dir=source_dir,
        config_file=config_file,
        output_dir=output_dir,
        preview=True,
        executable_resolver=lambda name: f"/tools/{name}",
        runner=fake_runner,
    )

    target = output_dir / "01_example.svg"
    assert rendered == (target,)
    assert calls == [
        (
            [
                "/tools/mmdc",
                "-i",
                str(source),
                "-o",
                str(target),
                "-t",
                "neutral",
                "-c",
                str(config_file),
                "-b",
                "white",
            ],
            True,
        ),
        (
            ["/tools/chafa", "-f", "kitty", "--fit-width", str(target)],
            True,
        ),
    ]


def test_render_requires_mermaid_cli_before_creating_output(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "example.mmd").write_text("flowchart LR\n    A --> B\n", encoding="utf-8")
    output_dir = tmp_path / "rendered"
    config_file = tmp_path / "mermaid.config.json"
    config_file.write_text('{"htmlLabels": false}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="'mmdc' is required"):
        render_schema_diagrams(
            source_dir=source_dir,
            config_file=config_file,
            output_dir=output_dir,
            executable_resolver=lambda _name: None,
        )

    assert not output_dir.exists()
