from __future__ import annotations

from sqlalchemy import text


DELETE_EXAMPLE_DAGS_SQL = text(
    """
    DELETE FROM dag
    WHERE bundle_name = :bundle_name
       OR fileloc LIKE :example_fileloc
       OR relative_fileloc LIKE :example_relative_fileloc
    """
)


def cleanup_example_dags() -> int:
    from airflow.settings import Session

    with Session() as session:
        result = session.execute(
            DELETE_EXAMPLE_DAGS_SQL,
            {
                "bundle_name": "example_dags",
                "example_fileloc": "%/airflow/example_dags/%",
                "example_relative_fileloc": "%example_dags%",
            },
        )
        session.commit()

    return int(result.rowcount or 0)


def main() -> None:
    deleted = cleanup_example_dags()
    print(f"cleanup_airflow_examples=deleted_dags:{deleted}")


if __name__ == "__main__":
    main()
