import math

from app.chunking import chunk_rfq, rfq_header
from app.embeddings import HashEmbedder


def _rfq(description="Short description.", notes=""):
    return {
        "title": "Bracket",
        "material": "Aluminium 6061",
        "process": "CNC milling",
        "quantity": 100,
        "customer": "Acme Test GmbH",
        "customer_country": "DE",
        "description": description,
        "technical_notes": notes,
    }


def _body(chunk, rfq):
    h = rfq_header(rfq) + "\n"
    assert chunk.startswith(h)
    return chunk[len(h):]


def test_header_format():
    assert rfq_header(_rfq()) == "Bracket | Aluminium 6061 | CNC milling | qty 100 | Acme Test GmbH (DE)"


def test_header_in_every_chunk():
    para = " ".join(["word"] * 150)
    rfq = _rfq(notes="\n\n".join([para] * 3))
    chunks = chunk_rfq(rfq)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.startswith(rfq_header(rfq) + "\n")


def test_short_rfq_single_chunk():
    assert len(chunk_rfq(_rfq(notes="One note."))) == 1


def test_long_notes_split_under_limit():
    para = " ".join(["word"] * 150)
    rfq = _rfq(description=para, notes="\n\n".join([para] * 2))
    chunks = chunk_rfq(rfq)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(_body(c, rfq).split()) <= 220


def test_giant_paragraph_split_by_sentence():
    sentence = " ".join(["alpha"] * 19) + "."
    giant = " ".join([sentence] * 30)  # 600 words
    rfq = _rfq(description=giant)
    chunks = chunk_rfq(rfq)
    assert len(chunks) >= 3
    for c in chunks:
        body = _body(c, rfq)
        assert len(body.split()) <= 220
        assert body.rstrip().endswith(".")


def test_hash_embedder_unit_norm():
    v = HashEmbedder().embed_query("CNC aluminium bracket")
    assert len(v) == 768
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6


def test_hash_embedder_similarity():
    e = HashEmbedder()
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))
    q = e.embed_query("CNC aluminium bracket")
    near = e.embed_passages(["aluminium bracket CNC milled"])[0]
    far = e.embed_passages(["plastic gear"])[0]
    assert cos(q, near) > cos(q, far)
