import json
import sys
from pathlib import Path

import psycopg
import pytest

from app.chunking import chunk_rfq
from app.data.generate import generate_rfqs
from app.data.scale import chunk_stats, generate_scale_rfqs
from app.embeddings import HashEmbedder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import scale_bench as sb  # noqa: E402


# ---------------------------------------------------------------- generator
def _is_long(r):
    return len(r["technical_notes"].split("\n\n")) == 5


def test_deterministic_and_small_config():
    a = generate_scale_rfqs(n_batches=2, batch_size=100)
    b = generate_scale_rfqs(n_batches=2, batch_size=100)
    assert a == b
    assert [r["rfq_id"] for r in a][:2] == ["RFQ-S00001", "RFQ-S00002"]
    assert generate_scale_rfqs(n_batches=2, batch_size=100, long_seed=1) != a


def test_full_count_unique_ids_and_long_share():
    rfqs = generate_scale_rfqs()
    assert len(rfqs) == 20000
    assert len({r["rfq_id"] for r in rfqs}) == 20000
    n_long = sum(_is_long(r) for r in rfqs)
    assert 0.13 < n_long / 20000 < 0.17
    # the exact count follows from the seeded rng: one draw per RFQ, plus draws for long ones
    assert n_long == sum(1 for r in generate_scale_rfqs() if _is_long(r))


def test_long_notes_make_multi_chunk_rfqs():
    rfqs = generate_scale_rfqs(n_batches=3, batch_size=400)
    stats = chunk_stats(rfqs)
    n_long = sum(_is_long(r) for r in rfqs)
    assert stats["rfqs"] == 1200
    assert stats["multi_chunk_rfqs"] >= 0.9 * n_long > 0
    assert stats["max_chunks_per_rfq"] >= 2
    assert stats["chunks"] == sum(len(chunk_rfq(r)) for r in rfqs)
    # short RFQs (original notes, at most 3 paragraphs) mostly stay single-chunk
    assert stats["multi_chunk_rfqs"] < 0.3 * len(rfqs)


def test_long_notes_formatted():
    for r in generate_scale_rfqs(n_batches=1, batch_size=400):
        assert "{" not in r["technical_notes"]


def test_seed_42_output_unchanged():
    path = ROOT / "data" / "rfqs.jsonl"
    if not path.exists():
        pytest.skip("data/rfqs.jsonl missing")
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert generate_rfqs(seed=42, n=400) == saved


# ---------------------------------------------------------------- helpers
def test_recall_at_k():
    assert sb.recall_at_k([1, 2, 3], [1, 2, 3]) == 1.0
    assert sb.recall_at_k([1, 9, 8], [1, 2, 3]) == pytest.approx(1 / 3)
    assert sb.recall_at_k([3, 2, 1, 7], [1, 2], k=2) == pytest.approx(0.5)
    assert sb.recall_at_k([], []) == 1.0


def test_percentile():
    assert sb.percentile([], 50) == 0.0
    assert sb.percentile([5.0], 95) == 5.0
    assert sb.percentile([1, 2, 3, 4], 50) == 2.5
    assert sb.percentile([4, 1, 3, 2], 100) == 4
    assert sb.latency_stats([1, 2, 3])["mean_ms"] == 2


def test_swap_db_url():
    assert sb.swap_db_url("postgresql://rag:rag@db:5432/rag", "rag_scale") == \
        "postgresql://rag:rag@db:5432/rag_scale"
    assert sb.swap_db_url("postgresql://u:p@127.0.0.1:5432/rag?sslmode=disable", "x") == \
        "postgresql://u:p@127.0.0.1:5432/x?sslmode=disable"


def test_collapse_to_rfq():
    assert sb.collapse_to_rfq(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert sb.collapse_to_rfq([]) == []


def test_stratified_sample():
    flags = [i % 5 == 0 for i in range(1000)]  # 20 % multi
    idx = sb.stratified_sample(flags, 100, seed=1)
    assert len(idx) == len(set(idx)) == 100
    assert sum(flags[i] for i in idx) == 20
    assert idx == sb.stratified_sample(flags, 100, seed=1)
    assert sb.stratified_sample(flags, 5000, seed=1) == list(range(1000))
    few = sb.stratified_sample([True] * 3 + [False] * 50, 10, seed=2)
    assert len(few) == 10


def test_rfq_batches_never_split_rfqs():
    rfqs = generate_scale_rfqs(n_batches=1, batch_size=400)
    batches = list(sb.rfq_batches(rfqs, max_chunks=32))
    assert [r["rfq_id"] for b in batches for r, _ in b] == [r["rfq_id"] for r in rfqs]
    assert all(sum(len(cs) for _, cs in b) <= 32 for b in batches)


# ---------------------------------------------------------------- integration
@pytest.mark.integration
def test_loader_resume_index_and_bench():
    from tests.conftest import _test_url

    url = sb.swap_db_url(_test_url(), "rag_scale_test")
    admin = sb.swap_db_url(url, "rag")
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as c:
            c.execute("DROP DATABASE IF EXISTS rag_scale_test WITH (FORCE)")
    except psycopg.OperationalError:
        pytest.skip("database not reachable")
    emb = HashEmbedder()
    rfqs = generate_scale_rfqs(n_batches=2, batch_size=400)
    try:
        sb.ensure_db(url)
        conn = sb.open_scale_conn(url)
        assert not sb.index_exists(conn)
        # resume: first load part of the corpus, then the full corpus
        part = sb.ingest_corpus(conn, rfqs[:300], emb, commit_every=2, log=lambda *_: None)
        assert part["rfqs_loaded"] == 300
        full = sb.ingest_corpus(conn, rfqs, emb, commit_every=2, log=lambda *_: None)
        assert full["rfqs_present_before"] == 300 and full["rfqs_loaded"] == 500
        n_rfq, n_chunk = conn.execute(
            "SELECT (SELECT count(*) FROM rfqs), (SELECT count(*) FROM chunks)").fetchone()
        assert n_rfq == 800 and n_chunk == chunk_stats(rfqs)["chunks"]
        assert sb.build_index(conn, "64MB") > 0
        assert sb.index_exists(conn)
        assert sb.db_sizes(conn, "rag_scale_test")[sb.INDEX]["relation_bytes"] > 0
        conn.close()

        from app.data.generate import generate_queries
        from app.db import connect
        texts = [q["text"] for q in generate_queries(seed=19, n=8)]
        vecs = [emb.embed_query(t) for t in texts]
        conn = connect(url)
        res = sb.run_bench(conn, texts, vecs, ef_grid=(10, 40, 400), warmup=2,
                           assert_default_plan=False, log=lambda *_: None)
        assert "Seq Scan" in res["plan_exact"] and sb.INDEX not in res["plan_exact"]
        assert res["plan_default"]
        assert [r["ef_search"] for r in res["ef_rows"]] == [10, 40, 400] * 2
        assert [r["planner"] for r in res["ef_rows"]] == ["default"] * 3 + ["seqscan off"] * 3
        assert all(r["scan"] == "hnsw" for r in res["ef_rows"][3:])
        assert res["shipped_depth_scan"] in ("hnsw", "seq scan")
        assert "::vector" not in sb.short_plan("x '[0.1,0.2]'::vector y").replace("'[...]'::vector", "")
        assert all(0.0 <= r["recall_at_10"] <= 1.0 for r in res["ef_rows"])
        assert 0.0 <= res["app_level"]["recall_at_10"] <= 1.0
        md = sb.render_md({
            "commit": "x", "timestamp": "t", "corpus": chunk_stats(rfqs),
            "config": {"n_queries": 8, "query_seed": 19, "warmup": 2}, **res}, None, None)
        assert "ef_search" in md
        conn.close()
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute("DROP DATABASE IF EXISTS rag_scale_test WITH (FORCE)")
