import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from app.config import get_settings
from app.db import connect, init_schema


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs a running Postgres+pgvector database"
    )


def _test_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    parts = urlsplit(get_settings().database_url)
    return urlunsplit(parts._replace(path="/rag_test"))


def open_test_conn():
    """Connect to the rag_test DB (created if missing) with the schema applied."""
    url = _test_url()
    try:
        admin = urlsplit(url)._replace(path="/rag")
        with psycopg.connect(
            urlunsplit(admin), autocommit=True, connect_timeout=3
        ) as c:
            exists = c.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'rag_test'"
            ).fetchone()
            if not exists:
                c.execute("CREATE DATABASE rag_test")
        conn = connect(url, register=False)
        init_schema(conn)
    except psycopg.OperationalError:
        pytest.skip("database not reachable")
    return conn


@pytest.fixture
def db_conn():
    conn = open_test_conn()
    yield conn
    conn.rollback()
    conn.close()
