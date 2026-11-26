import argparse
import json
import time

from app.chunking import chunk_rfq
from app.db import connect, init_schema
from app.embeddings import Embedder

BATCH = 32
RFQ_COLS = [
    "rfq_id", "date", "customer", "customer_industry", "customer_country",
    "language", "title", "material", "process", "quantity", "status", "family_id",
]


def ingest(conn, rfqs: list[dict], embedder: Embedder) -> tuple[int, int]:
    """Replace the corpus in one transaction. Returns (n_rfqs, n_chunks)."""
    pending = [(r["rfq_id"], i, c) for r in rfqs for i, c in enumerate(chunk_rfq(r))]
    try:
        conn.execute("TRUNCATE rfqs, chunks RESTART IDENTITY CASCADE")
        placeholders = ", ".join(["%s"] * len(RFQ_COLS))
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO rfqs ({', '.join(RFQ_COLS)}) VALUES ({placeholders})",
                [[r[c] for c in RFQ_COLS] for r in rfqs],
            )
            for s in range(0, len(pending), BATCH):
                batch = pending[s : s + BATCH]
                vecs = embedder.embed_passages([c for _, _, c in batch])
                cur.executemany(
                    "INSERT INTO chunks (rfq_id, chunk_index, content, embedding)"
                    " VALUES (%s, %s, %s, %s)",
                    [(rid, i, c, v) for (rid, i, c), v in zip(batch, vecs)],
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rfqs), len(pending)


def main() -> None:
    from app.embeddings import E5Embedder

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/rfqs.jsonl")
    args = ap.parse_args()
    with open(args.path, encoding="utf-8") as f:
        rfqs = [json.loads(line) for line in f if line.strip()]
    t0 = time.time()
    conn = connect(register=False)
    init_schema(conn)
    n_rfqs, n_chunks = ingest(conn, rfqs, E5Embedder())
    print(f"Ingested {n_rfqs} RFQs, {n_chunks} chunks in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
