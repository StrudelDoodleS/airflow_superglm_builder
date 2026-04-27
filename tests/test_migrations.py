from pathlib import Path

from pricing_pipeline.migrations import migration_files, split_sql_server_batches


def test_split_sql_server_batches_handles_go_lines():
    sql = "SELECT 1;\nGO\nSELECT 2;\ngo\n"
    assert split_sql_server_batches(sql) == ["SELECT 1;", "SELECT 2;"]


def test_migration_files_are_sorted(tmp_path: Path):
    (tmp_path / "V002__b.sql").write_text("SELECT 2", encoding="utf-8")
    (tmp_path / "V001__a.sql").write_text("SELECT 1", encoding="utf-8")
    assert [p.name for p in migration_files(tmp_path)] == ["V001__a.sql", "V002__b.sql"]
