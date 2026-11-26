"""Dense, lexical and hybrid (RRF) retrieval over the chunk table."""
import re
from dataclasses import asdict, dataclass
from typing import Literal

import psycopg

DEPTH = 50

META_COLS = "r.rfq_id, r.title, r.customer, r.customer_industry, r.customer_country, r.status, r.date"


@dataclass
class SearchHit:
    rfq_id: str
    title: str
    customer: str
    customer_industry: str
    customer_country: str
    status: str
    date: str
    score: float
    chunk_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def rrf(
    ranked_lists: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """(Weighted) reciprocal rank fusion. Sorted by score desc, ties by first appearance."""
    weights = weights if weights is not None else [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])  # stable: dict order = first appearance


DENSE_CANDIDATE_SQL = """
SELECT rfq_id, content, embedding <=> %(v)s::vector AS dist
FROM chunks
ORDER BY dist
LIMIT %(cand)s
"""


HNSW_EF_SEARCH_MAX = 1000  # pgvector's hard cap for hnsw.ef_search


def dense_candidates(conn, embedder, query: str, depth: int) -> list[tuple[str, float, str]]:
    vec = embedder.embed_query(query)
    cand = depth * 4
    # Index-friendly ORDER BY ... LIMIT (HNSW), collapsed to best chunk per RFQ in Python.
    # An HNSW scan returns at most ef_search rows (default 40), so raise it to the candidate
    # count. set_config(..., is_local=true) is SET LOCAL, which lasts until the surrounding
    # transaction ends; with autocommit off and a long-lived connection (eval scripts) that
    # would be "never". So the work runs in a transaction block (a savepoint if the caller
    # already has a transaction open) that is rolled back at the end: the query is read-only,
    # and the rollback undoes the setting. (No SHOW/restore: hnsw.ef_search does not exist in
    # a session until pgvector's library has been loaded.)
    # enable_seqscan = off: pgvector 0.8.7 costs an HNSW scan higher as ef_search grows, so at
    # ~13k RFQs the planner picked a seq scan + sort for depth 50. Disabling seq scans keeps the
    # HNSW index whenever it exists; without an index Postgres still falls back to a seq scan.
    # The transaction rollback undoes this setting too.
    ef = min(cand, HNSW_EF_SEARCH_MAX)
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(ef)])
        conn.execute("SET LOCAL enable_seqscan = off")
        rows = conn.execute(DENSE_CANDIDATE_SQL, {"v": vec, "cand": cand}).fetchall()
        raise psycopg.Rollback()
    best: dict[str, tuple[str, float, str]] = {}
    for rid, content, dist in rows:  # rows arrive in ascending distance
        if rid not in best:
            best[rid] = (rid, 1.0 - float(dist), content)
    return list(best.values())[:depth]


def _lexical_query(conn, tsquery_sql: str, params: list, depth: int):
    rows = conn.execute(
        f"""
        SELECT rfq_id, score, content FROM (
            SELECT DISTINCT ON (rfq_id) rfq_id, content,
                   ts_rank_cd(tsv, q) AS score
            FROM chunks, ({tsquery_sql}) AS qq(q)
            WHERE tsv @@ q
            ORDER BY rfq_id, ts_rank_cd(tsv, q) DESC
        ) best
        ORDER BY score DESC, rfq_id
        LIMIT %s
        """,
        [*params, depth],
    ).fetchall()
    return [(r[0], float(r[1]), r[2]) for r in rows]


def lexical_candidates(conn, query: str, depth: int) -> list[tuple[str, float, str]]:
    rows = _lexical_query(
        conn, "SELECT websearch_to_tsquery('simple', %s)", [query], depth
    )
    if rows:
        return rows
    tokens = list(dict.fromkeys(re.findall(r"[\w.-]+", query.lower())))
    if not tokens:
        return []
    # One plainto_tsquery per token (bound parameter), OR-ed together.
    sql = "SELECT " + " || ".join(["plainto_tsquery('simple', %s)"] * len(tokens))
    return _lexical_query(conn, sql, tokens, depth)


def _hydrate(conn, ranked: list[tuple[str, float]], chunks: dict[str, str]) -> list[SearchHit]:
    if not ranked:
        return []
    ids = [rid for rid, _ in ranked]
    meta = {
        row[0]: row
        for row in conn.execute(
            f"SELECT {META_COLS} FROM rfqs r WHERE r.rfq_id = ANY(%s)", [ids]
        ).fetchall()
    }
    hits = []
    for rid, score in ranked:
        m = meta[rid]
        hits.append(
            SearchHit(
                rfq_id=m[0], title=m[1], customer=m[2], customer_industry=m[3],
                customer_country=m[4], status=m[5], date=str(m[6]),
                score=score, chunk_text=chunks[rid],
            )
        )
    return hits


def rerank_ranked(
    reranker, query: str, ranked: list[tuple[str, float]], chunks: dict[str, str], depth: int
) -> list[tuple[str, float]]:
    """Score the first `depth` candidates with `reranker`; stable sort by score desc
    (ties keep first-stage order). The returned score is the reranker score."""
    cand = ranked[:depth]
    if not cand:
        return []
    scores = reranker.score(query, [chunks[rid] for rid, _ in cand])
    if len(scores) != len(cand):
        raise ValueError("reranker returned a wrong number of scores")
    pairs = [(rid, float(s)) for (rid, _), s in zip(cand, scores)]
    return sorted(pairs, key=lambda kv: -kv[1])


def search(
    conn,
    embedder,
    query: str,
    k: int = 5,
    mode: Literal["dense", "lexical", "hybrid"] = "hybrid",
    lexical_weight: float = 1.0,
    rrf_k: int = 60,
    reranker=None,
    rerank_depth: int | None = None,
) -> list[SearchHit]:
    rerank = reranker is not None and rerank_depth is not None
    if rerank and not 1 <= rerank_depth <= DEPTH:
        raise ValueError(f"rerank_depth must be between 1 and {DEPTH}")
    use_lexical = mode == "lexical" or (mode == "hybrid" and lexical_weight != 0)
    dense = dense_candidates(conn, embedder, query, DEPTH) if mode in ("dense", "hybrid") else []
    lexical = lexical_candidates(conn, query, DEPTH) if use_lexical else []
    chunks = {rid: text for rid, _, text in dense}
    chunks.update({rid: text for rid, _, text in lexical if rid not in chunks})
    if mode == "dense":
        ranked = [(rid, s) for rid, s, _ in dense]
    elif mode == "lexical":
        ranked = [(rid, s) for rid, s, _ in lexical]
    else:
        ranked = rrf(
            [[r[0] for r in dense], [r[0] for r in lexical]],
            k=rrf_k, weights=[1.0, lexical_weight],
        )
    if rerank:
        # rerank at least k candidates so a small RERANK_DEPTH never returns fewer than k hits
        ranked = rerank_ranked(reranker, query, ranked, chunks, max(rerank_depth, min(k, DEPTH)))
    return _hydrate(conn, ranked[:k], chunks)
