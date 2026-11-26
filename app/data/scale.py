"""Scale corpus: many seeded batches of `generate_rfqs`, a share of them with long notes.

Pure and deterministic. Protocol fixed before the run; summarised in the README scale-test section.
"""
import random

from app.chunking import chunk_rfq
from app.data.families import FAMILIES
from app.data.generate import NOTES, _fmt_int, generate_rfqs

_FAMILY = {f.family_id: f for f in FAMILIES}


def _long_notes(rng: random.Random, rfq: dict) -> str:
    """Five paragraphs from two independently shuffled copies of the language's note pool."""
    lang = rfq["language"]
    ctx = {
        "weeks": rng.randint(2, 4),
        "demand": _fmt_int(rfq["quantity"] * rng.randint(2, 6), lang),
        "s1": _FAMILY[rfq["family_id"]].standards[0],
    }
    pool = []
    for _ in range(2):
        copy = list(NOTES[lang])
        rng.shuffle(copy)
        pool.extend(copy)
    per_par = len(pool) // 5
    return "\n\n".join(
        " ".join(s.format(**ctx) for s in pool[i * per_par:(i + 1) * per_par]) for i in range(5)
    )


def generate_scale_rfqs(
    n_batches: int = 50,
    batch_size: int = 400,
    seed_base: int = 1000,
    long_share: float = 0.15,
    long_seed: int = 2026,
) -> list[dict]:
    rows: list[dict] = []
    for b in range(n_batches):
        rows.extend(generate_rfqs(seed=seed_base + b, n=batch_size))
    rng = random.Random(long_seed)
    out = []
    for i, r in enumerate(rows, start=1):
        r = {**r, "rfq_id": f"RFQ-S{i:05d}"}
        if rng.random() < long_share:
            r["technical_notes"] = _long_notes(rng, r)
        out.append(r)
    return out


def chunk_stats(rfqs: list[dict]) -> dict:
    counts = [len(chunk_rfq(r)) for r in rfqs]
    return {
        "rfqs": len(rfqs),
        "chunks": sum(counts),
        "multi_chunk_rfqs": sum(1 for c in counts if c > 1),
        "max_chunks_per_rfq": max(counts, default=0),
    }
