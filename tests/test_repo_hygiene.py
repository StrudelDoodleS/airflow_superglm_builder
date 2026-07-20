from pathlib import Path


def test_zip_archives_are_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "*.zip" in gitignore
    assert "state/" in gitignore


def test_project_package_can_be_imported():
    import pricing_pipeline

    assert pricing_pipeline.__version__


def test_removed_validation_workaround_modules_do_not_return():
    for path in (
        "pricing_pipeline/build_identity.py",
        "pricing_pipeline/modeling/superglm_identity.py",
        "pricing_pipeline/modeling/validation_curves.py",
    ):
        assert not Path(path).exists()
