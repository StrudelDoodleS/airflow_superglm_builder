from pathlib import Path


def test_zip_archives_are_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "*.zip" in gitignore
    assert "state/" in gitignore


def test_project_package_can_be_imported():
    import pricing_pipeline

    assert pricing_pipeline.__version__
