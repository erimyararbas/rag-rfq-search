"""Cited answers: Claude answers over retrieved RFQs using search_result blocks."""
from dataclasses import dataclass
from typing import Any

from app.retrieval import SearchHit

SYSTEM_PROMPT = (
    "You help a manufacturing sales engineer triage a new request for quotation. "
    "Use only the past RFQs provided as search results. Say which past RFQs are most "
    "similar, which customers requested them and with what status, and point out "
    "differences that matter for quoting (material, process, quantity, tolerances). "
    "If none of the results is a reasonable match, say so plainly. Keep it under 200 words."
)

REFUSAL_ANSWER = "The model declined to answer this request."
SOURCE_PREFIX = "rfq://"
TRUNCATION_NOTE = "\n\n[Answer truncated: max_tokens reached.]"


@dataclass
class Citation:
    rfq_id: str
    cited_text: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]


def build_search_results(hits: list[SearchHit]) -> list[dict]:
    return [
        {
            "type": "search_result",
            "source": f"{SOURCE_PREFIX}{h.rfq_id}",
            "title": f"{h.rfq_id} – {h.title} – {h.customer}",
            "content": [{"type": "text", "text": h.chunk_text}],
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def _citation_rfq_id(cit: Any, hits: list[SearchHit]) -> str | None:
    idx = getattr(cit, "search_result_index", None)
    if isinstance(idx, int) and 0 <= idx < len(hits):
        return hits[idx].rfq_id
    # Fallback: parse the source URI, but only accept ids we actually sent.
    source = getattr(cit, "source", None) or ""
    if source.startswith(SOURCE_PREFIX):
        rfq_id = source[len(SOURCE_PREFIX):]
        if any(h.rfq_id == rfq_id for h in hits) and idx is None:
            return rfq_id
    return None


def answer_query(client: Any, model: str, query: str, hits: list[SearchHit]) -> AnswerResult:
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [*build_search_results(hits), {"type": "text", "text": query}],
            }
        ],
    )
    if response.stop_reason == "refusal":
        return AnswerResult(REFUSAL_ANSWER, [])

    parts: list[str] = []
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for cit in getattr(block, "citations", None) or []:
            rfq_id = _citation_rfq_id(cit, hits)
            if rfq_id is None:
                continue
            key = (rfq_id, cit.cited_text)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(rfq_id, cit.cited_text))
    answer = "".join(parts)
    if response.stop_reason == "max_tokens":
        answer += TRUNCATION_NOTE
    return AnswerResult(answer, citations)
