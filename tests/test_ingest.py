import pytest

from app.data.generate import generate_rfqs
from app.embeddings import HashEmbedder
from app.ingest import ingest

pytestmark = pytest.mark.integration


def _count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_ingest_counts(db_conn):
    rfqs = generate_rfqs(42, 10)
    n_rfqs, n_chunks = ingest(db_conn, rfqs, HashEmbedder())
    assert n_rfqs == 10
    assert n_chunks >= 10
    assert _count(db_conn, "rfqs") == 10
    assert _count(db_conn, "chunks") == n_chunks


def test_ingest_idempotent(db_conn):
    rfqs = generate_rfqs(42, 10)
    first = ingest(db_conn, rfqs, HashEmbedder())
    second = ingest(db_conn, rfqs, HashEmbedder())
    assert first == second
    assert _count(db_conn, "chunks") == second[1]


def test_tsv_populated(db_conn):
    ingest(db_conn, generate_rfqs(42, 10), HashEmbedder())
    empty = db_conn.execute(
        "SELECT count(*) FROM chunks WHERE tsv IS NULL OR tsv = ''::tsvector"
    ).fetchone()[0]
    assert empty == 0
