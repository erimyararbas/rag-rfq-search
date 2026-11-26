import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from app.data.code_queries import code_pattern, generate_code_queries, relevant_ids, rfq_text
from app.data.generate import generate_queries, generate_rfqs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_code_queries  # noqa: E402


@pytest.fixture(scope="module")
def rfqs():
    return generate_rfqs(42)


@pytest.fixture(scope="module")
def queries(rfqs):
    return generate_code_queries(rfqs)


@pytest.mark.parametrize("code,text,expected", [
    ("1.4301", "in 1.4301.", True),
    ("1.4301", "1.4301, ISO", True),
    ("1.4301", "Werkstoff 1.4301", True),
    ("1.4301", "1.43012", False),
    ("1.4301", "11.4301", False),
    ("1.4301", "1.4301-2", False),
    ("1.4301", "x.1.4301", False),
    ("1.4301", "1.4301.5", False),
    ("ISO 286-1", "per ISO 286-1.", True),
    ("ISO 286-1", "per ISO 286-12", False),
    ("ISO 286-1", "per iso 286-1,", True),
    ("42CrMo4", "steel 42crmo4 bar", True),
    ("42CrMo4", "x42CrMo4", False),
])
def test_boundary_rule(code, text, expected):
    assert bool(code_pattern(code).search(text)) is expected


def test_rfq_text_fields():
    r = {"title": "T", "description": "D", "technical_notes": "N"}
    assert rfq_text(r) == "T D N"


def test_relevant_ids_requires_all_codes():
    rfqs = [
        {"rfq_id": "A", "title": "1.4301", "description": "ISO 1101", "technical_notes": ""},
        {"rfq_id": "B", "title": "1.4301", "description": "", "technical_notes": ""},
        {"rfq_id": "C", "title": "", "description": "ISO 1101", "technical_notes": ""},
    ]
    assert relevant_ids(rfqs, ["1.4301"]) == ["A", "B"]
    assert relevant_ids(rfqs, ["1.4301", "ISO 1101"]) == ["A"]


def test_deterministic(rfqs, queries):
    assert generate_code_queries(rfqs) == queries


def test_counts_and_languages(queries):
    assert len(queries) == 60
    assert Counter(q["type"] for q in queries) == {"single": 20, "pair": 20, "sentence": 20}
    assert Counter(q["language"] for q in queries if q["type"] == "sentence") == {
        "en": 8, "de": 6, "tr": 6}
    assert all(q["language"] == "code" for q in queries if q["type"] != "sentence")
    assert [q["query_id"] for q in queries] == [f"C-{i:03d}" for i in range(1, 61)]


def test_relevance_matches_recompute(rfqs, queries):
    for q in queries:
        assert 1 <= q["n_relevant"] <= 30
        assert q["n_relevant"] == len(q["relevant"])
        assert q["relevant"] == relevant_ids(rfqs, q["codes"])


def test_unique_codes_per_type_and_text(queries):
    for t in ("single", "pair", "sentence"):
        tuples = [tuple(q["codes"]) for q in queries if q["type"] == t]
        assert len(tuples) == len(set(tuples)) == 20
    for q in queries:
        assert all(c in q["text"] for c in q["codes"])
        assert len(q["codes"]) == (1 if q["type"] == "single" else 2)


def test_on_disk_equals_fresh(queries):
    path = ROOT / "eval" / "code_queries.jsonl"
    on_disk = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert on_disk == queries


def test_existing_generators_unchanged(rfqs):
    generate_code_queries(rfqs)
    assert generate_rfqs(42) == rfqs
    assert generate_queries(7) == generate_queries(7)


def test_eval_rank_by_scores_stable():
    assert eval_code_queries.rank_by_scores(["a", "b", "c"], [0.5, 0.9, 0.5]) == ["b", "a", "c"]
    with pytest.raises(ValueError):
        eval_code_queries.rank_by_scores(["a"], [])
