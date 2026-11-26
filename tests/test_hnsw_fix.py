import sys
from pathlib import Path

import pytest

from app.data.generate import generate_rfqs
from app.embeddings import HashEmbedder
from app.ingest import ingest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_hnsw_fix as ch  # noqa: E402


def _row(scan="hnsw", r50=0.99):
    return {"scan": scan, "plan": "Limit\n  ->  Index Scan using chunks_embedding_hnsw on chunks",
            "recall_at_50": r50, "recall_at_10": 1.0, "mean_ms": 3.0, "p50_ms": 2.5, "p95_ms": 5.0}


def _large(after):
    return {"db_rfqs": 13200, "n_queries": 200, "query_seed": 29, "warmup": 10,
            "before": _row("seq scan", 1.0), "after": after}


def test_count_identical_requires_same_order():
    assert ch.count_identical([["a", "b"], ["a", "b"], ["a"]], [["a", "b"], ["b", "a"], ["a"]]) == 2


def test_large_pass_criteria():
    assert ch.large_passes(_large(_row("hnsw", 0.98)))
    assert not ch.large_passes(_large(_row("hnsw", 0.979)))
    assert not ch.large_passes(_large(_row("seq scan", 1.0)))


def test_small_pass_criteria():
    assert ch.small_passes({"identical": 140, "total": 140})
    assert not ch.small_passes({"identical": 139, "total": 140})


def test_merge_keeps_other_sections():
    out = ch.merge_results({"small": {"x": 1}, "commit": "old"}, "large", {"y": 2})
    assert out["small"] == {"x": 1} and out["large"] == {"y": 2}
    assert out["commit"] != "old" and "timestamp" in out


def test_render_md_sections_and_no_private_path():
    small = {"db_rfqs": 400, "total": 140, "n_family": 80, "n_code": 60, "identical": 140, "scan": "hnsw"}
    md = ch.render_md({"commit": "abc", "timestamp": "t", "small": small})
    assert "140 / 140" in md and "PASS" in md and "Check 2" not in md
    md = ch.render_md({"commit": "abc", "timestamp": "t", "small": small, "large": _large(_row("seq scan", 1.0))})
    assert "| before | seq scan |" in md and "| after | seq scan |" in md
    assert "FAIL" in md
    assert "docs/superpowers" not in md and "protocol fixed before the run" in md.lower()


@pytest.mark.integration
def test_run_small_and_large_on_test_db(db_conn):
    emb = HashEmbedder()
    ingest(db_conn, generate_rfqs(seed=42, n=40), emb)
    texts = ["aluminium housing milled", "steel bracket welded", "plastic cap injection"]
    vecs = [emb.embed_query(t) for t in texts]
    s = ch.run_small(db_conn, texts, vecs, n_family=2)
    assert s["total"] == 3 and s["n_code"] == 1 and 0 <= s["identical"] <= 3
    assert s["scan"] == "hnsw" and s["db_rfqs"] == 40
    g = ch.run_large(db_conn, texts, vecs, warmup=1, log=lambda *_: None)
    assert g["after"]["scan"] == "hnsw"
    assert 0.0 <= g["before"]["recall_at_50"] <= 1.0 and g["after"]["mean_ms"] > 0
    assert db_conn.execute("SHOW enable_seqscan").fetchone()[0] == "on"
