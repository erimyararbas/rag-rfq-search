from types import SimpleNamespace

import pytest

from app.answer import SYSTEM_PROMPT, answer_query, build_search_results
from app.retrieval import SearchHit


def _hit(rfq_id: str) -> SearchHit:
    return SearchHit(
        rfq_id=rfq_id, title="Bracket", customer="Acme GmbH", customer_industry="auto",
        customer_country="DE", status="won", date="2024-01-01", score=0.5,
        chunk_text=f"chunk of {rfq_id}",
    )


HITS = [_hit("RFQ-0001"), _hit("RFQ-0002")]


def _cite(idx, text="cited"):
    return SimpleNamespace(
        type="search_result_location", source=f"rfq://RFQ-000{idx + 1}", title="t",
        cited_text=text, search_result_index=idx, start_block_index=0, end_block_index=1,
    )


def _response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


class FakeClient:
    def __init__(self, response=None, exc=None):
        self.kwargs = None
        self._response, self._exc = response, exc
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self._exc:
            raise self._exc
        return self._response


def _default_response():
    return _response([
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text="Most similar is ", citations=None),
        SimpleNamespace(type="text", text="RFQ-0002.", citations=[_cite(1)]),
    ])


def test_request_shape():
    client = FakeClient(_default_response())
    answer_query(client, "claude-opus-5-5", "need a bracket", HITS)
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == SYSTEM_PROMPT
    assert "thinking" not in kw
    content = kw["messages"][0]["content"]
    assert content[:2] == build_search_results(HITS)
    assert all(b["type"] == "search_result" and b["citations"]["enabled"] is True for b in content[:2])
    assert content[0]["source"] == "rfq://RFQ-0001"
    assert content[-1] == {"type": "text", "text": "need a bracket"}


def test_citation_mapping():
    res = answer_query(FakeClient(_default_response()), "m", "q", HITS)
    assert res.answer == "Most similar is RFQ-0002."
    assert [(c.rfq_id, c.cited_text) for c in res.citations] == [("RFQ-0002", "cited")]


def test_out_of_range_citation_dropped():
    resp = _response([SimpleNamespace(type="text", text="Answer.", citations=[_cite(9), _cite(0)])])
    res = answer_query(FakeClient(resp), "m", "q", HITS)
    assert res.answer == "Answer."
    assert [c.rfq_id for c in res.citations] == ["RFQ-0001"]


def test_duplicate_citations_deduped():
    resp = _response([SimpleNamespace(type="text", text="A.", citations=[_cite(0), _cite(0)])])
    assert len(answer_query(FakeClient(resp), "m", "q", HITS).citations) == 1


def test_refusal():
    resp = _response([SimpleNamespace(type="text", text="x", citations=[_cite(0)])], stop_reason="refusal")
    res = answer_query(FakeClient(resp), "m", "q", HITS)
    assert res.answer == "The model declined to answer this request."
    assert res.citations == []


def test_max_tokens_appends_truncation_note_and_keeps_citations():
    resp = _response(
        [SimpleNamespace(type="text", text="Most similar is RFQ-0002", citations=[_cite(1)])],
        stop_reason="max_tokens",
    )
    result = answer_query(FakeClient(resp), "m", "q", HITS)
    assert result.answer == "Most similar is RFQ-0002\n\n[Answer truncated: max_tokens reached.]"
    assert [c.rfq_id for c in result.citations] == ["RFQ-0002"]
