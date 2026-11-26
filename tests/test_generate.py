import os
from collections import Counter

from app.data.generate import generate_queries, generate_rfqs

KEYS = {
    "rfq_id", "date", "customer", "customer_industry", "customer_country",
    "language", "title", "description", "technical_notes", "material",
    "process", "quantity", "status", "family_id",
}

DENYLIST = ["sk-ant"]
DENYLIST += [w.strip().lower() for w in os.environ.get("RAG_DENYLIST", "").split(",") if w.strip()]


def test_deterministic():
    assert generate_rfqs(42) == generate_rfqs(42)
    assert generate_rfqs(42) != generate_rfqs(43)


def test_shape():
    rows = generate_rfqs(42)
    assert len(rows) == 400
    assert len({r["rfq_id"] for r in rows}) == 400
    for r in rows:
        assert KEYS <= set(r)
        assert r["quantity"] > 0
        assert r["status"] in {"quoted", "won", "lost"}
        assert r["language"] in {"en", "de", "tr"}
        assert "2023-01-01" <= r["date"] <= "2025-12-31"
        assert r["description"].strip()


def test_language_mix():
    rows = generate_rfqs(42)
    share = {k: v / len(rows) for k, v in Counter(r["language"] for r in rows).items()}
    assert 0.5 <= share["en"] <= 0.7
    assert 0.15 <= share["de"] <= 0.35
    assert 0.08 <= share["tr"] <= 0.22


def test_family_coverage():
    rows = generate_rfqs(42)
    per_family = Counter(r["family_id"] for r in rows)
    assert len(per_family) >= 35
    assert min(per_family.values()) >= 5


def test_customers_and_notes():
    rows = generate_rfqs(42)
    per_customer = Counter(r["customer"] for r in rows)
    assert 1 <= min(per_customer.values()) and max(per_customer.values()) <= 15
    long_notes = [r for r in rows if r["technical_notes"].count("\n\n") >= 1]
    assert 0.2 <= len(long_notes) / len(rows) <= 0.4  # exercises chunking


def test_denylist():
    blobs = [
        " ".join(str(v) for v in r.values()).lower()
        for r in generate_rfqs(42) + generate_queries(7)
    ]
    for word in DENYLIST:
        assert not any(word in b for b in blobs), word


def test_queries_disjoint():
    rows = generate_rfqs(42)
    queries = generate_queries(7)
    assert len(queries) == 80
    descriptions = {r["description"] for r in rows}
    assert all(q["text"] not in descriptions for q in queries)
    families = {r["family_id"] for r in rows}
    assert all(q["family_id"] in families for q in queries)
    majority = {}
    for fam in families:
        langs = Counter(r["language"] for r in rows if r["family_id"] == fam)
        majority[fam] = langs.most_common(1)[0][0]
    cross = sum(q["language"] != majority[q["family_id"]] for q in queries)
    assert cross / len(queries) >= 0.30


def _words(text):
    import re
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) >= 4}


def test_queries_use_paraphrases():
    """Most queries must not reuse the corpus part-name vocabulary."""
    from app.data.families import FAMILIES, PROC
    from app.data.query_vocab import GENERIC_MATERIAL

    # process / material words are shared on purpose, so they do not count as name overlap
    shared = set()
    for words in list(PROC.values()) + list(GENERIC_MATERIAL.values()):
        for w in words:
            shared |= _words(w)
    fams = {f.family_id: f for f in FAMILIES}
    queries = generate_queries(7)
    disjoint = 0
    for q in queries:
        name_words = _words(" ".join(fams[q["family_id"]].names[q["language"]])) - shared
        if not name_words & _words(q["text"]):
            disjoint += 1
    assert disjoint / len(queries) >= 0.5
