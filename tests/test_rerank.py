import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.data.generate import generate_rfqs
from app.embeddings import HashEmbedder
from app.ingest import ingest
from app.main import create_app
from app.retrieval import rerank_ranked, search
from tests.conftest import open_test_conn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_rerank  # noqa: E402

QUERY = "Aluminiumgehäuse gefräst 6061"


class FakeReranker:
    """Score = number of distinct query tokens that occur in the passage."""

    def __init__(self):
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        toks = set(re.findall(r"\w+", query.lower()))
        return [float(len(toks & set(re.findall(r"\w+", p.lower())))) for p in passages]


def test_rerank_ranked_orders_by_score():
    chunks = {"a": "x", "b": "x y", "c": "x y z"}
    out = rerank_ranked(FakeReranker(), "x y z", [("a", 9.0), ("b", 8.0), ("c", 7.0)], chunks, 3)
    assert out == [("c", 3.0), ("b", 2.0), ("a", 1.0)]


def test_rerank_ranked_ties_keep_first_stage_order():
    chunks = {"a": "q", "b": "q", "c": "q"}
    out = rerank_ranked(FakeReranker(), "q", [("a", 3.0), ("b", 2.0), ("c", 1.0)], chunks, 3)
    assert [r for r, _ in out] == ["a", "b", "c"]


def test_rerank_ranked_depth_cut_and_empty():
    chunks = {"a": "q", "b": "q", "c": "q"}
    out = rerank_ranked(FakeReranker(), "q", [("a", 3.0), ("b", 2.0), ("c", 1.0)], chunks, 2)
    assert [r for r, _ in out] == ["a", "b"]
    assert rerank_ranked(FakeReranker(), "q", [], {}, 5) == []


# ---- integration ----

RFQS = generate_rfqs(seed=42, n=400)


@pytest.fixture(scope="module")
def ready():
    conn = open_test_conn()
    ingest(conn, RFQS, HashEmbedder())
    conn.close()
    return True


@pytest.mark.integration
def test_search_with_reranker(ready):
    conn = open_test_conn()
    try:
        fake = FakeReranker()
        hits = search(conn, HashEmbedder(), QUERY, k=7, mode="dense", reranker=fake, rerank_depth=30)
        ids = [h.rfq_id for h in hits]
        assert 0 < len(ids) <= 7 and len(ids) == len(set(ids))
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(float(s).is_integer() for s in scores)  # reranker scores, not cosine
        assert fake.calls == 1
        few = search(conn, HashEmbedder(), QUERY, k=10, mode="dense", reranker=fake, rerank_depth=3)
        assert len(few) == 10  # depth below k still returns k hits
        for bad in (0, 51, -1):
            with pytest.raises(ValueError):
                search(conn, HashEmbedder(), QUERY, reranker=fake, rerank_depth=bad)
    finally:
        conn.close()


@pytest.mark.integration
def test_search_without_rerank_unchanged(ready):
    conn = open_test_conn()
    try:
        a = search(conn, HashEmbedder(), QUERY, k=5, mode="hybrid", lexical_weight=1.0)
        b = search(conn, HashEmbedder(), QUERY, k=5, mode="hybrid", lexical_weight=1.0,
                   reranker=FakeReranker())  # no depth -> no rerank
        assert [h.to_dict() for h in a] == [h.to_dict() for h in b]
    finally:
        conn.close()


@pytest.mark.integration
def test_api_rerank_factory_called_once(ready):
    made = []

    def factory():
        made.append(1)
        return FakeReranker()

    app = create_app(open_test_conn, HashEmbedder(), None,
                     Settings(rerank_depth=30, _env_file=None), reranker_factory=factory)
    c = TestClient(app)
    for _ in range(2):
        r = c.post("/search", json={"query": QUERY, "k": 5, "rerank": True})
        assert r.status_code == 200
        assert 0 < len(r.json()["results"]) <= 5
    assert made == [1]


@pytest.mark.integration
def test_api_rerank_without_factory_503(ready):
    app = create_app(open_test_conn, HashEmbedder(), None, Settings(_env_file=None))
    r = TestClient(app).post("/search", json={"query": "aluminium", "rerank": True})
    assert r.status_code == 503
    assert r.json() == {"detail": "reranker not configured"}


@pytest.mark.integration
def test_api_rerank_omitted_same_results(ready):
    settings = Settings(_env_file=None)
    plain = TestClient(create_app(open_test_conn, HashEmbedder(), None, settings))
    with_f = TestClient(create_app(open_test_conn, HashEmbedder(), None, settings,
                                   reranker_factory=FakeReranker))
    body = {"query": QUERY, "k": 5}
    expected = plain.post("/search", json=body).json()
    assert with_f.post("/search", json=body).json() == expected
    assert with_f.post("/search", json={**body, "rerank": False}).json() == expected


# ---- eval script helpers ----

def test_selection_key_tiebreak():
    a = {"depth": 20, "dev": {"rr": 0.5, "hit@5": 0.6}}
    b = {"depth": 30, "dev": {"rr": 0.5, "hit@5": 0.6}}
    c = {"depth": 50, "dev": {"rr": 0.5, "hit@5": 0.7}}
    d = {"depth": 50, "dev": {"rr": 0.4, "hit@5": 0.9}}
    assert max([b, a], key=eval_rerank.selection_key)["depth"] == 20
    assert max([a, b, c], key=eval_rerank.selection_key)["depth"] == 50
    assert max([a, d], key=eval_rerank.selection_key)["depth"] == 20


def test_percentile():
    assert eval_rerank.percentile([], 50) == 0.0
    assert eval_rerank.percentile([5.0], 95) == 5.0
    assert eval_rerank.percentile([1, 2, 3, 4], 50) == 2.5
    assert eval_rerank.percentile([4, 1, 3, 2], 100) == 4
    assert eval_rerank.percentile(list(range(101)), 95) == 95


def test_rank_by_scores_stable():
    assert eval_rerank.rank_by_scores(["a", "b", "c", "d"], [1, 2, 2, 0]) == ["b", "c", "a", "d"]
    with pytest.raises(ValueError):
        eval_rerank.rank_by_scores(["a"], [])
