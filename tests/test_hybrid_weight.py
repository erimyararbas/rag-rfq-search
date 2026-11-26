import pytest

from app.data.generate import generate_rfqs
from app.embeddings import HashEmbedder
from app.ingest import ingest
from app.retrieval import search

pytestmark = pytest.mark.integration


def test_search_lexical_weight_zero_equals_dense_order(db_conn):
    emb = HashEmbedder()
    ingest(db_conn, generate_rfqs(seed=42, n=100), emb)
    q = "aluminium housing machining tolerance"
    dense = [h.rfq_id for h in search(db_conn, emb, q, k=10, mode="dense")]
    hybrid = [h.rfq_id for h in search(db_conn, emb, q, k=10, mode="hybrid", lexical_weight=0.0)]
    assert dense and hybrid == dense
