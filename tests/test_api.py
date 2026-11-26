import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.data.generate import generate_rfqs
from app.embeddings import HashEmbedder
from app.ingest import ingest
from app.main import create_app
from tests.conftest import open_test_conn

pytestmark = pytest.mark.integration

RFQS = generate_rfqs(seed=42, n=400)


@pytest.fixture(scope="module")
def client():
    conn = open_test_conn()
    embedder = HashEmbedder()
    ingest(conn, RFQS, embedder)
    conn.close()
    app = create_app(open_test_conn, embedder, None, Settings(hybrid_lexical_weight=1.0, _env_file=None))
    yield TestClient(app)


def _post(client, **body):
    return client.post("/search", json=body)


def test_material_and_process_lexical(client):
    r = RFQS[0]
    assert r["rfq_id"] == "RFQ-0001"
    resp = _post(client, query=f"{r['material']} {r['process']}", mode="lexical")
    assert resp.status_code == 200
    assert "RFQ-0001" in [h["rfq_id"] for h in resp.json()["results"]][:5]


def test_or_token_fallback_when_and_query_matches_nothing(client):
    r = RFQS[0]
    resp = _post(client, query=f"{r['material']} zzqxnonexistent", mode="lexical")
    assert resp.status_code == 200
    assert resp.json()["results"] != []


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_modes_return_k(client, mode):
    resp = _post(client, query="Aluminiumgehäuse gefräst 6061", k=7, mode=mode)
    assert resp.status_code == 200
    ids = [h["rfq_id"] for h in resp.json()["results"]]
    assert 0 < len(ids) <= 7
    assert len(ids) == len(set(ids))


def test_no_lexical_match_falls_back(client):
    lex = _post(client, query="???", mode="lexical")
    hyb = _post(client, query="???", mode="hybrid")
    assert lex.status_code == 200 and hyb.status_code == 200
    assert lex.json()["results"] == []
    assert hyb.json()["results"] != []


def test_empty_query_422(client):
    assert _post(client, query="   ").status_code == 422


def test_health_counts(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["rfqs"] == 400
    assert body["chunks"] >= 400


def test_ask_without_key_503(client):
    resp = client.post("/ask", json={"query": "aluminium housing"})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "ANTHROPIC_API_KEY not set; /search still works"}


def _fake_claude(exc=None):
    from types import SimpleNamespace

    def create(**kwargs):
        if exc:
            raise exc
        cite = SimpleNamespace(
            type="search_result_location", source="x", title="t", cited_text="evidence",
            search_result_index=0, start_block_index=0, end_block_index=1,
        )
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text="Best match.", citations=[cite]),
            ],
            stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_ask_with_fake_client(client):
    app = create_app(open_test_conn, HashEmbedder(), _fake_claude(), Settings(hybrid_lexical_weight=1.0, _env_file=None))
    resp = TestClient(app).post("/ask", json={"query": "Aluminiumgehäuse gefräst 6061", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Best match."
    source_ids = {s["rfq_id"] for s in body["sources"]}
    assert len(body["sources"]) == 3
    assert body["citations"] and all(c["rfq_id"] in source_ids for c in body["citations"])


def test_ask_upstream_error_502(client):
    import anthropic
    import httpx

    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    app = create_app(open_test_conn, HashEmbedder(), _fake_claude(exc=err), Settings(hybrid_lexical_weight=1.0, _env_file=None))
    resp = TestClient(app).post("/ask", json={"query": "anything"})
    assert resp.status_code == 502
    assert resp.json() == {"detail": "upstream model error"}


def test_ask_without_hits_does_not_call_claude(client, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Claude must not be called without hits")

    from types import SimpleNamespace

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr("app.main.search", lambda *a, **k: [])
    app = create_app(open_test_conn, HashEmbedder(), fake, Settings(_env_file=None))
    resp = TestClient(app).post("/ask", json={"query": "anything"})
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": "No similar past RFQs were found.", "citations": [], "sources": [],
    }
    assert calls == []


def test_index_page_served():
    app = create_app(open_test_conn, HashEmbedder(), None, Settings(_env_file=None))
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    for needle in ("RFQ similarity search", "/search", "/ask", "API key not configured"):
        assert needle in resp.text
