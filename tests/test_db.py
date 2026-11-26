import psycopg
import pytest
from urllib.parse import urlsplit, urlunsplit

from app.db import connect
from tests.conftest import _test_url

pytestmark = pytest.mark.integration


def test_connect_without_extension_raises_clear_error():
    base = urlsplit(_test_url())
    admin_url = urlunsplit(base._replace(path="/rag"))
    url = urlunsplit(base._replace(path="/rag_noext"))
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("database not reachable")
    with admin:
        admin.execute("DROP DATABASE IF EXISTS rag_noext")
        admin.execute("CREATE DATABASE rag_noext")
        try:
            with pytest.raises(RuntimeError, match="run ingest first"):
                connect(url)
            with psycopg.connect(url) as c:  # connect() must not have created it
                assert c.execute(
                    "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
                ).fetchone()[0] == 0
        finally:
            admin.execute("DROP DATABASE IF EXISTS rag_noext")
