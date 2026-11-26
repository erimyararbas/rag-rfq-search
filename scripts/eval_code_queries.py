"""Pre-registered exact-code query eval: five systems, nothing tuned on this set."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.db import connect  # noqa: E402
from app.evaluation import _agg  # noqa: E402
from app.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k  # noqa: E402
from app.retrieval import DEPTH, rerank_ranked, search  # noqa: E402

K_RETRIEVED = 20
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_NAME = "rerank (default top 50)"
SYSTEMS = ["dense", "lexical", "hybrid (equal weights)", "shipped default", RERANK_NAME]
TYPES = ("single", "pair", "sentence")

PROTOCOL = """\
Protocol (fixed before any run on this query set; not changed after seeing results)
- Question: does lexical retrieval or hybrid fusion help when the query contains exact material codes or standard numbers?
- Codes: material grades (material field) and standard designations that appear in the RFQ text. Thread sizes and tolerances are excluded.
- Relevance: an RFQ is relevant to a query if its text (title, description, technical notes) contains every code of the query as an exact, case-insensitive match with token boundaries (code not preceded by a word character, '.' or '-'; not followed by a word character, '-', or '.' plus a digit). Computed from data/rfqs.jsonl at generation time and stored in the query file; family_id is not used.
- Query set: eval/code_queries.jsonl, generator seed 23. 60 queries, 20 each of single (one bare code), pair (material + standard, bare), sentence (the pair in a request sentence; 8 en, 6 de, 6 tr). Kept only if 1 to 30 relevant RFQs and a new code tuple within its type.
- Systems (nothing tuned on this set): dense; lexical; hybrid with equal weights (w_lex = 1.0, k = 60); shipped default (hybrid with the settings defaults, w_lex = 0.0, equal to dense); rerank = shipped default top 50 reranked with BAAI/bge-reranker-v2-m3 (max_length 512, CPU) at the depth already selected on the family dev split (50).
- Metrics: hit@5, P@5, R@10, MRR@20 (top 20 retrieved), mean n_relevant. Every query is reported; there is no split.
- Latency: mean per system; rerank = first stage plus the reranker call (timed after one untimed warm-up).
- Results are reported whatever they show. README wording about exact codes is updated from these numbers only.
"""


def git_commit(arg):
    if arg or os.environ.get("GIT_COMMIT"):
        return arg or os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def rank_by_scores(candidates: list[str], scores: list[float]) -> list[str]:
    """Candidate ids sorted by score desc; stable, so ties keep first-stage order."""
    if len(candidates) != len(scores):
        raise ValueError("scores and candidates differ in length")
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order]


def metric_row(ranked: list[str], relevant: set[str]) -> dict:
    return {
        "n_relevant": len(relevant),
        "hit@5": hit_rate_at_k(ranked, relevant, 5),
        "p@5": precision_at_k(ranked, relevant, 5),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "rr": mrr(ranked, relevant),
    }


def table(title: str, systems: list[tuple[str, dict]]) -> list[str]:
    lines = [title, "", "| system | hit@5 | P@5 | R@10 | MRR@20 |", "|---|---|---|---|---|"]
    for name, a in systems:
        lines.append(
            f"| {name} | {a['hit@5']:.3f} | {a['p@5']:.3f} | {a['recall@10']:.3f} | {a['rr']:.3f} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None)
    ap.add_argument("--queries", default=str(ROOT / "eval" / "code_queries.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    import torch

    from app.embeddings import E5Embedder
    from app.rerank import CrossEncoderReranker

    queries = [json.loads(line) for line in
               Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    conn = connect()
    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model)
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(RERANK_MODEL, max_length=512)
    load_s = time.perf_counter() - t0
    threads = torch.get_num_threads()
    depth = settings.rerank_depth
    assert 1 <= depth <= DEPTH

    configs = {
        "dense": dict(mode="dense", lexical_weight=1.0, rrf_k=60),
        "lexical": dict(mode="lexical", lexical_weight=1.0, rrf_k=60),
        "hybrid (equal weights)": dict(mode="hybrid", lexical_weight=1.0, rrf_k=60),
        "shipped default": dict(mode="hybrid", lexical_weight=settings.hybrid_lexical_weight,
                                rrf_k=settings.rrf_k),
    }

    def run(q, cfg, k):
        t0 = time.perf_counter()
        hits = search(conn, embedder, q["text"], k=k, **cfg)
        return hits, (time.perf_counter() - t0) * 1000

    # one untimed warm-up of the reranker
    w_hits, _ = run(queries[0], configs["shipped default"], DEPTH)
    reranker.score(queries[0]["text"], [h.chunk_text for h in w_hits[:depth]])

    per_query = []
    for q in queries:
        relevant = set(q["relevant"])
        base = {"query_id": q["query_id"], "type": q["type"], "language": q["language"]}
        for name, cfg in configs.items():
            hits, ms = run(q, cfg, K_RETRIEVED)
            per_query.append({"system": name, **base,
                              **metric_row([h.rfq_id for h in hits], relevant), "latency_ms": ms})
        hits, fs_ms = run(q, configs["shipped default"], DEPTH)
        ranked = [(h.rfq_id, h.score) for h in hits]
        chunks = {h.rfq_id: h.chunk_text for h in hits}
        t0 = time.perf_counter()
        reranked = rerank_ranked(reranker, q["text"], ranked, chunks, depth)
        rr_ms = (time.perf_counter() - t0) * 1000
        ids = [rid for rid, _ in reranked][:K_RETRIEVED]
        per_query.append({"system": RERANK_NAME, **base, **metric_row(ids, relevant),
                          "latency_ms": fs_ms + rr_ms, "reranker_ms": rr_ms})

    def agg_for(pred):
        return [(n, _agg([r for r in per_query if r["system"] == n and pred(r)])) for n in SYSTEMS]

    def mean_rel(qs):
        return sum(q["n_relevant"] for q in qs) / len(qs) if qs else 0.0

    md_parts = [PROTOCOL]
    md_parts += table(f"All queries (n = {len(queries)}; mean n_relevant {mean_rel(queries):.2f})",
                      agg_for(lambda r: True)) + [""]
    for t in TYPES:
        qs = [q for q in queries if q["type"] == t]
        md_parts += table(f"Type: {t} (n = {len(qs)}; mean n_relevant {mean_rel(qs):.2f})",
                          agg_for(lambda r, t=t: r["type"] == t)) + [""]
    for lang in ("en", "de", "tr"):
        qs = [q for q in queries if q["type"] == "sentence" and q["language"] == lang]
        if not qs:
            continue
        md_parts += table(
            f"Sentence queries, language {lang} (n = {len(qs)}; mean n_relevant {mean_rel(qs):.2f})",
            agg_for(lambda r, lang=lang: r["type"] == "sentence" and r["language"] == lang)) + [""]
    lat = {}
    lat_lines = ["Mean latency per query", "", "| system | mean ms |", "|---|---|"]
    for n in SYSTEMS:
        vals = [r["latency_ms"] for r in per_query if r["system"] == n]
        lat[n] = sum(vals) / len(vals) if vals else 0.0
        lat_lines.append(f"| {n} | {lat[n]:.1f} |")
    rr_vals = [r["reranker_ms"] for r in per_query if "reranker_ms" in r]
    rr_mean = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0
    md_parts += lat_lines + ["", f"Reranker call alone (depth {depth}): mean {rr_mean:.1f} ms. "
                                 f"Reranker load time: {load_s:.1f} s. torch threads: {threads}.", ""]
    md = "\n".join(md_parts)

    result = {
        "protocol": PROTOCOL,
        "queries_file": Path(args.queries).name,
        "systems": {**configs, RERANK_NAME: {
            "first_stage": configs["shipped default"], "first_stage_k": DEPTH,
            "rerank_depth": depth, "reranker_model": RERANK_MODEL, "max_length": 512}},
        "k_retrieved": K_RETRIEVED,
        "mean_latency_ms": lat, "reranker_call_mean_ms": rr_mean,
        "model_load_seconds": load_s, "torch_threads": threads,
        "embedding_model": settings.embedding_model,
        "per_query": per_query,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(args.commit),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "code_queries.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "code_queries.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
