"""Render committed SQL Mermaid diagrams and optionally preview them in Kitty."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "sql" / "diagrams"
CONFIG_FILE = SOURCE_DIR / "mermaid.config.json"
DEFAULT_OUTPUT_DIR = ROOT / "state" / "db_diagrams"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered SVG files. Defaults to state/db_diagrams.",
    )
    parser.add_argument(
        "--theme",
        choices=("default", "forest", "dark", "neutral"),
        default="neutral",
        help="Mermaid render theme.",
    )
    parser.add_argument(
        "--background-color",
        default="white",
        help="SVG background color. Defaults to white for terminal readability.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview each rendered SVG using chafa -f kitty --fit-width.",
    )
    return parser


def _required_executable(
    name: str,
    resolver: Callable[[str], str | None],
) -> str:
    executable = resolver(name)
    if executable is None:
        raise RuntimeError(
            f"{name!r} is required but was not found on PATH. "
            "Install Mermaid CLI for mmdc and Chafa for terminal previews."
        )
    return executable


def render_schema_diagrams(
    *,
    source_dir: Path = SOURCE_DIR,
    config_file: Path = CONFIG_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    theme: str = "neutral",
    background_color: str = "white",
    preview: bool = False,
    executable_resolver: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., object] = subprocess.run,
) -> tuple[Path, ...]:
    sources = sorted(source_dir.glob("*.mmd"))
    if not sources:
        raise RuntimeError(f"No Mermaid diagram sources found in {source_dir}")
    if not config_file.is_file():
        raise RuntimeError(f"Mermaid configuration does not exist: {config_file}")

    mmdc = _required_executable("mmdc", executable_resolver)
    chafa = _required_executable("chafa", executable_resolver) if preview else None
    resolved_output = output_dir if output_dir.is_absolute() else ROOT / output_dir
    resolved_output.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    for source in sources:
        target = resolved_output / f"{source.stem}.svg"
        runner(
            [
                mmdc,
                "-i",
                str(source),
                "-o",
                str(target),
                "-t",
                theme,
                "-c",
                str(config_file),
                "-b",
                background_color,
            ],
            check=True,
        )
        print(f"rendered={target}")
        rendered.append(target)
        if chafa is not None:
            runner(
                [chafa, "-f", "kitty", "--fit-width", str(target)],
                check=True,
            )
    return tuple(rendered)


def main() -> None:
    args = build_parser().parse_args()
    try:
        render_schema_diagrams(
            output_dir=args.output_dir,
            theme=args.theme,
            background_color=args.background_color,
            preview=args.preview,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
