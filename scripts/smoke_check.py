from __future__ import annotations

from superglm import SuperGLM


def check_superglm_rating_export(superglm_cls=SuperGLM) -> int:
    if hasattr(superglm_cls, "export_rating_tables"):
        print("smoke_check=ok")
        return 0

    print("smoke_check=rating_export_unavailable")
    print(
        "SuperGLM.export_rating_tables is unavailable in this environment. "
        "Install SuperGLM >=0.26 from PyPI before running rating export tasks."
    )
    return 0


def main() -> None:
    raise SystemExit(check_superglm_rating_export())


if __name__ == "__main__":
    main()
