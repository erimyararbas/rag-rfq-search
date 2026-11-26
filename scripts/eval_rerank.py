"""Pre-registered rerank experiment: cross-encoder depth selected on dev, reported on held-out test."""
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
from app.embeddings import E5Embedder  # noqa: E402
from app.evaluation import (  # noqa: E402
    _agg, is_cross_language, majority_languages, split_families,
)
from app.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k  # noqa: E402
from app.retrieval import search  # noqa: E402

SPLIT_SEED = 11
DEPTH_GRID = (20, 30, 50)
K_FIRST_STAGE = 50
K_RETRIEVED = 20
BASELINE = "baseline (first stage)"

PROTOCOL = """\
Protocol (fixed before any rerank run; not changed after seeing results)
- Queries and corpus: the existing 80 queries (seed 7) and 400 RFQs (seed 42), unchanged.
- Split: by family, split_families with seed 11 (the same dev and test families as the hybrid sweep).
- First stage: the shipped default, hybrid with the settings defaults (w_lex = 0.0, k = 60, so it equals dense), top 50.
- Reranker input: query text and the best chunk text of each candidate, as stored in the DB. No instruction prefix. Model BAAI/bge-reranker-v2-m3, max_length 512, CPU.
- Depth grid: {20, 30, 50}. For each depth d the reranker actually scores the first d candidates (a separate, timed call per depth). The reranked top d forms the final ranking; metrics are computed on its top 20 (for d = 20 rerank only reorders the top 20).
- Selection: dev MRR@20. Tie-break: higher dev hit@5, then smaller depth.
- Reported: baseline and every depth on dev and on test; the full set (80) with same-language and cross-language subsets, labelled as including the dev half. Test numbers are reported whatever they show. No second selection round.
- Latency: one untimed warm-up call, then per query the wall time of the reranker call per depth (mean, p50, p95), plus first-stage latency. Model load time and torch.get_num_threads() are recorded. CPU only.
- Product decision fixed in advance: rerank stays opt-in (rerank=true) whatever the result, because of its CPU latency. RERANK_DEPTH is set to the selected depth.
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


def depth_name(d: int) -> str:
    return f"rerank depth {d}"


def selection_key(r: dict):
    # max(): higher dev MRR@20, higher dev hit@5, then smaller depth
    return (r["dev"]["rr"], r["dev"]["hit@5"], -r["depth"])


def percentile(vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100); 0.0 for empty input."""
    if not vals:
        return 0.0
    s = sorted(vals)
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


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


def latency_stats(vals: list[float]) -> dict:
    return {
        "mean": sum(vals) / len(vals) if vals else 0.0,
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
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
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    import torch

    from app.rerank import CrossEncoderReranker

    queries = [json.loads(line) for line in
               Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    conn = connect()
    fam_rows = conn.execute("SELECT family_id, language, rfq_id FROM rfqs").fetchall()
    family_rfqs: dict[str, set[str]] = {}
    for fam, _, rid in fam_rows:
        family_rfqs.setdefault(fam, set()).add(rid)
    fam_lang = majority_languages([(f, lang) for f, lang, _ in fam_rows])
    dev_fams, test_fams = split_families({q["family_id"] for q in queries}, seed=SPLIT_SEED)

    settings = get_settings()
    embedder = E5Embedder(settings.embedding_model)
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3", max_length=512)
    load_s = time.perf_counter() - t0
    threads = torch.get_num_threads()

    # First stage once per query (timed), then an untimed warm-up of the reranker.
    first = {}
    for q in queries:
        t0 = time.perf_counter()
        hits = search(
            conn, embedder, q["text"], k=K_FIRST_STAGE, mode="hybrid",
            lexical_weight=settings.hybrid_lexical_weight, rrf_k=settings.rrf_k,
        )
        first[q["query_id"]] = (
            [h.rfq_id for h in hits], {h.rfq_id: h.chunk_text for h in hits},
            (time.perf_counter() - t0) * 1000,
        )
    w_ids, w_chunks, _ = first[queries[0]["query_id"]]
    reranker.score(queries[0]["text"], [w_chunks[i] for i in w_ids[:DEPTH_GRID[0]]])

    per_query = []
    for q in queries:
        ids, chunks, fs_ms = first[q["query_id"]]
        relevant = family_rfqs[q["family_id"]]
        base = {
            "query_id": q["query_id"], "family_id": q["family_id"], "language": q["language"],
            "split": "dev" if q["family_id"] in dev_fams else "test",
            "cross_language": is_cross_language(q["language"], fam_lang[q["family_id"]]),
        }
        per_query.append({
            "system": BASELINE, **base, **metric_row(ids[:K_RETRIEVED], relevant),
            "latency_ms": fs_ms,
        })
        for d in DEPTH_GRID:
            cand = ids[:d]
            t0 = time.perf_counter()
            scores = reranker.score(q["text"], [chunks[i] for i in cand])
            ms = (time.perf_counter() - t0) * 1000
            ranked = rank_by_scores(cand, scores)[:K_RETRIEVED]
            per_query.append({
                "system": depth_name(d), "depth": d, **base, **metric_row(ranked, relevant),
                "latency_ms": ms,
            })

    systems = [BASELINE] + [depth_name(d) for d in DEPTH_GRID]

    def rows_of(system, pred=lambda r: True):
        return [r for r in per_query if r["system"] == system and pred(r)]

    grid = [{"depth": d, "dev": _agg(rows_of(depth_name(d), lambda r: r["split"] == "dev"))}
            for d in DEPTH_GRID]
    best = max(grid, key=selection_key)

    def agg_for(pred):
        return [(n, _agg(rows_of(n, pred))) for n in systems]

    n_dev = sum(1 for q in queries if q["family_id"] in dev_fams)
    n_cross = len({r["query_id"] for r in per_query if r["cross_language"]})
    lat_lines = ["| stage | mean ms | p50 ms | p95 ms |", "|---|---|---|---|"]
    fs = latency_stats([r["latency_ms"] for r in rows_of(BASELINE)])
    lat_lines.append(f"| first stage (hybrid, top 50) | {fs['mean']:.1f} | {fs['p50']:.1f} | {fs['p95']:.1f} |")
    latency = {"first_stage": fs}
    for d in DEPTH_GRID:
        st = latency_stats([r["latency_ms"] for r in rows_of(depth_name(d))])
        latency[depth_name(d)] = st
        lat_lines.append(f"| rerank depth {d} | {st['mean']:.1f} | {st['p50']:.1f} | {st['p95']:.1f} |")

    md = "\n".join(
        [PROTOCOL, f"Dev: {len(dev_fams)} families, {n_dev} queries. Test: {len(test_fams)} families, "
                   f"{len(queries) - n_dev} queries.", "",
         f"Selected depth: {best['depth']} (dev MRR@20 {best['dev']['rr']:.3f}, dev hit@5 {best['dev']['hit@5']:.3f}).", ""]
        + table("Dev split (selection)", agg_for(lambda r: r["split"] == "dev")) + [""]
        + table("Test split (held out)", agg_for(lambda r: r["split"] == "test")) + [""]
        + table(f"Full set (n = {len(queries)}; includes the dev half that selected the depth)",
                agg_for(lambda r: True)) + [""]
        + table("Full set, same-language queries (includes the dev half)",
                agg_for(lambda r: not r["cross_language"])) + [""]
        + table(f"Full set, cross-language queries (n = {n_cross}; includes the dev half)",
                agg_for(lambda r: r["cross_language"])) + [""]
        + ["Latency (per query; reranker on CPU)", ""] + lat_lines + [
            "", f"Reranker load time: {load_s:.1f} s. torch threads: {threads}.", ""]
    )

    result = {
        "protocol": PROTOCOL, "split_seed": SPLIT_SEED,
        "dev_families": sorted(dev_fams), "test_families": sorted(test_fams),
        "first_stage": {"mode": "hybrid", "k": K_FIRST_STAGE,
                        "lexical_weight": settings.hybrid_lexical_weight, "rrf_k": settings.rrf_k},
        "reranker_model": "BAAI/bge-reranker-v2-m3", "max_length": 512,
        "depth_grid": list(DEPTH_GRID),
        "dev_grid_results": [{"depth": g["depth"], **g["dev"]} for g in grid],
        "selected": {"depth": best["depth"]},
        "latency_ms": latency,
        "model_load_seconds": load_s, "torch_threads": threads,
        "embedding_model": settings.embedding_model,
        "per_query": per_query,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(args.commit),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rerank.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "rerank.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
