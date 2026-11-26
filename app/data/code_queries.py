"""Exact-code query set (pre-registered protocol, restated in eval/results/code_queries.md).

Codes are material grades and standard designations that occur in the RFQ text.
Relevance is exact, case-insensitive matching with token boundaries, computed
from the corpus itself (never from `family_id`).
"""

import random
import re

from app.data.families import FAMILIES

SEED = 23
N_PER_TYPE = 20
MIN_RELEVANT = 1
MAX_RELEVANT = 30
SENTENCE_LANGS = ["en"] * 8 + ["de"] * 6 + ["tr"] * 6
TYPES = ("single", "pair", "sentence")

TEMPLATES = {
    "en": ["Earlier RFQs with {material} and {standard}?",
           "Need a quote for parts in {material} according to {standard}."],
    "de": ["Frühere Anfragen mit {material} nach {standard}?",
           "Angebot für Teile aus {material} gemäß {standard}."],
    "tr": ["{material} malzemeli ve {standard} içeren eski talepler?",
           "{standard} standardına uygun {material} parça teklifi."],
}

_FAMILY_STANDARDS = {f.family_id: list(f.standards) for f in FAMILIES}


def code_pattern(code: str) -> re.Pattern:
    """Not preceded by a word char, '.' or '-'; not followed by a word char, '-',
    or '.' plus a digit. Case-insensitive."""
    return re.compile(r"(?<![\w.\-])" + re.escape(code) + r"(?![\w\-]|\.\d)", re.IGNORECASE)


def rfq_text(rfq: dict) -> str:
    return " ".join([rfq["title"], rfq["description"], rfq["technical_notes"]])


def relevant_ids(rfqs: list[dict], codes) -> list[str]:
    pats = [code_pattern(c) for c in codes]
    return sorted(r["rfq_id"] for r in rfqs
                  if all(p.search(rfq_text(r)) for p in pats))


def _standards_in_text(rfq: dict) -> list[str]:
    text = rfq_text(rfq)
    return [s for s in _FAMILY_STANDARDS[rfq["family_id"]] if code_pattern(s).search(text)]


def generate_code_queries(rfqs: list[dict], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    order = list(rfqs)
    rng.shuffle(order)
    texts = {r["rfq_id"]: rfq_text(r) for r in rfqs}
    cache: dict[tuple, list[str]] = {}

    def rel(codes: tuple) -> list[str]:
        if codes not in cache:
            pats = [code_pattern(c) for c in codes]
            cache[codes] = sorted(i for i, t in texts.items() if all(p.search(t) for p in pats))
        return cache[codes]

    langs = list(SENTENCE_LANGS)
    rng.shuffle(langs)

    out: list[dict] = []
    for qtype in TYPES:
        seen: set[tuple] = set()
        kept = 0
        for rfq in order:
            if kept == N_PER_TYPE:
                break
            stds = _standards_in_text(rfq)
            material = rfq["material"]
            if not code_pattern(material).search(texts[rfq["rfq_id"]]):
                continue
            if qtype == "single":
                if kept % 2 == 0:
                    codes = (material,)
                elif stds:
                    codes = (rng.choice(stds),)
                else:
                    continue
            else:
                if not stds:
                    continue
                codes = (material, rng.choice(stds))
            if codes in seen:
                continue
            relevant = rel(codes)
            if not MIN_RELEVANT <= len(relevant) <= MAX_RELEVANT:
                continue
            seen.add(codes)
            if qtype == "single":
                language, text = "code", codes[0]
            elif qtype == "pair":
                language, text = "code", f"{codes[0]} {codes[1]}"
            else:
                language = langs[kept]
                text = rng.choice(TEMPLATES[language]).format(material=codes[0], standard=codes[1])
            out.append({
                "query_id": f"C-{len(out) + 1:03d}", "type": qtype, "language": language,
                "text": text, "codes": list(codes), "relevant": relevant,
                "n_relevant": len(relevant),
            })
            kept += 1
    return out
