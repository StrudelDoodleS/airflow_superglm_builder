from __future__ import annotations

from superglm import SuperGLM


def main() -> None:
    assert hasattr(SuperGLM, "export_rating_tables")
    print("smoke_check=ok")


if __name__ == "__main__":
    main()
