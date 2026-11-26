import pytest

from app.data.generate import generate_rfqs
from app.embeddings import HashEmbedder
from app.ingest import ingest
from app.retrieval import DENSE_CANDIDATE_SQL, dense_candidates

pytestmark = pytest.mark.integration


def test_dense_query_can_use_hnsw_index(db_conn):
    """The query shape is servable by the HNSW index when the planner chooses it.

    At 400 rows the planner picks a sequential scan on its own, so the index is forced
    here with enable_seqscan = off; this checks the query *can* use it, not that it does.
    """
    emb = HashEmbedder()
    ingest(db_conn, generate_rfqs(seed=42, n=400), emb)
    try:
        db_conn.execute("SET LOCAL enable_seqscan = off")
        plan = "\n".join(
            r[0]
            for r in db_conn.execute(
                "EXPLAIN " + DENSE_CANDIDATE_SQL,
                {"v": emb.embed_query("aluminium"), "cand": 200},
            ).fetchall()
        )
    finally:
        db_conn.rollback()
    assert "chunks_embedding_hnsw" in plan, plan


def test_dense_depth_holds_when_hnsw_is_used(db_conn):
    """With the HNSW index forced, ef_search is raised so depth=50 still yields 50 RFQs."""
    emb = HashEmbedder()
    ingest(db_conn, generate_rfqs(seed=42, n=400), emb)
    try:
        db_conn.execute("SET LOCAL enable_seqscan = off")
        rows = dense_candidates(db_conn, emb, "aluminium housing milled", depth=50)
    finally:
        db_conn.rollback()
    ids = [r[0] for r in rows]
    assert len(ids) == 50 and len(set(ids)) == 50


def test_ef_search_does_not_leak_past_the_call(db_conn):
    emb = HashEmbedder()
    ingest(db_conn, generate_rfqs(seed=42, n=40), emb)
    before = db_conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    rows = dense_candidates(db_conn, emb, "aluminium", depth=50)
    after = db_conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    assert before == after == "40"
    assert rows
    # the planner setting must not leak either
    assert db_conn.execute("SHOW enable_seqscan").fetchone()[0] == "on"
    db_conn.rollback()


def test_dense_candidates_disables_seqscan_inside_the_call(db_conn, monkeypatch):
    """The fix: enable_seqscan is off while the dense query runs, on again afterwards."""
    emb = HashEmbedder()
    ingest(db_conn, generate_rfqs(seed=42, n=40), emb)
    seen = []
    real = db_conn.execute

    def spy(query, *a, **kw):
        if isinstance(query, str) and "chunks" in query and "ORDER BY" in query:
            seen.append(real("SHOW enable_seqscan").fetchone()[0])
        return real(query, *a, **kw)

    monkeypatch.setattr(db_conn, "execute", spy, raising=False)
    dense_candidates(db_conn, emb, "aluminium", depth=5)
    monkeypatch.undo()
    assert seen == ["off"]
    assert db_conn.execute("SHOW enable_seqscan").fetchone()[0] == "on"
    db_conn.rollback()


def test_dense_candidates_on_fresh_connection():
    """Regression: a brand-new session (pgvector library not yet loaded) must work."""
    from app.db import connect
    from tests.conftest import _test_url, open_test_conn

    open_test_conn().close()  # ensure schema exists
    conn = connect(_test_url())
    try:
        rows = dense_candidates(conn, HashEmbedder(), "aluminium", depth=5)
        assert len(rows) >= 1
    finally:
        conn.close()
