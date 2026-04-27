from pricing_pipeline.config import Settings
from pricing_pipeline.db import build_odbc_connect_string


def test_settings_defaults_are_local_dev_safe():
    settings = Settings.from_env({})
    assert settings.mssql_server == "mssql,1433"
    assert settings.pricing_database == "PricingLab"
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"


def test_odbc_connection_string_targets_database():
    settings = Settings.from_env(
        {
            "MSSQL_SERVER": "localhost,1433",
            "MSSQL_DATABASE": "PricingLab",
            "MSSQL_USER": "sa",
            "MSSQL_PASSWORD": "secret",
            "MSSQL_DRIVER": "ODBC Driver 18 for SQL Server",
            "MSSQL_ENCRYPT": "no",
            "MSSQL_TRUST_SERVER_CERT": "yes",
        }
    )
    odbc = build_odbc_connect_string(settings, database=settings.pricing_database)
    assert "SERVER=localhost,1433" in odbc
    assert "DATABASE=PricingLab" in odbc
    assert "PWD=secret" in odbc
