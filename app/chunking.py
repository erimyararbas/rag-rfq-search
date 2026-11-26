"""Split an RFQ into header-prefixed chunks that fit an embedding window."""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def rfq_header(rfq: dict) -> str:
    return (
        f"{rfq['title']} | {rfq['material']} | {rfq['process']} | "
        f"qty {rfq['quantity']} | {rfq['customer']} ({rfq['customer_country']})"
    )


def _wc(text: str) -> int:
    return len(text.split())


def _split_paragraph(par: str, max_words: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries (hard-split words as last resort)."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sent in _SENTENCE_SPLIT.split(par.strip()):
        n = _wc(sent)
        if n > max_words:
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            words = sent.split()
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i:i + max_words]))
            continue
        if count + n > max_words and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_rfq(rfq: dict, max_words: int = 220) -> list[str]:
    header = rfq_header(rfq) + "\n"
    paragraphs = [rfq.get("description") or ""] + (rfq.get("technical_notes") or "").split("\n\n")
    units: list[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        units.extend(_split_paragraph(p, max_words) if _wc(p) > max_words else [p])

    bodies: list[str] = []
    current: list[str] = []
    count = 0
    for u in units:
        n = _wc(u)
        if current and count + n > max_words:
            bodies.append("\n\n".join(current))
            current, count = [], 0
        current.append(u)
        count += n
    if current:
        bodies.append("\n\n".join(current))
    if not bodies:
        bodies = [""]
    return [header + b for b in bodies]
