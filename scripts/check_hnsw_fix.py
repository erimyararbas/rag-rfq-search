"""Acceptance checks for the HNSW planner fix (protocol fixed before the run).

Subcommands: small (400-RFQ `rag` DB: results identical to exact kNN), large (13,200-RFQ
`rag_scale` DB: the shipped depth-50 path uses the HNSW index, recall@50 >= 0.98).
Writes/merges eval/results/hnsw_fix.json and re-renders hnsw_fix.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import get_settings  # noqa: E402
from app.data.generate import generate_queries  # noqa: E402
from app.db import connect  # noqa: E402
from app.retrieval import DENSE_CANDIDATE_SQL, dense_candidates  # noqa: E402
from scale_bench import (  # noqa: E402
    OUT_DIR, SCALE_DB, PrecomputedEmbedder, _plan, collapse_to_rfq, exact_topk, git_commit,
    latency_stats, now_iso, read_json, recall_at_k, scan_node, short_plan, swap_db_url,
    write_json,
)

DEPTH = 50
EF = DEPTH * 4
WARMUP = 10
QUERY_SEED = 29
N_QUERIES = 200
MIN_RECALL = 0.98
JSON_NAME = "hnsw_fix.json"


# ---------------------------------------------------------------- pure helpers
def count_identical(got: list[list[str]], truth: list[list[str]]) -> int:
    """Queries whose RFQ id lists are equal (same ids, same order)."""
    return sum(1 for g, t in zip(got, truth) if list(g) == list(t))


def large_passes(large: dict) -> bool:
    a = large["after"]
    return a["scan"] == "hnsw" and a["recall_at_50"] >= MIN_RECALL


def small_passes(small: dict) -> bool:
    return small["identical"] == small["total"]


def merge_results(prev: dict | None, key: str, value: dict) -> dict:
    out = dict(prev or {})
    out[key] = value
    out["commit"], out["timestamp"] = git_commit(), now_iso()
    return out


def render_md(res: dict) -> str:
    L = ["# HNSW planner fix: acceptance checks", "",
         f"Commit `{res.get('commit', '?')}`, {res.get('timestamp', '?')}. Protocol fixed before the "
         "run. Change: `SET LOCAL enable_seqscan = off` inside the rolled-back transaction of "
         "`dense_candidates`, so the planner keeps the HNSW index when one exists.", ""]
    if "small" in res:
        s = res["small"]
        L += ["## Check 1: small corpus unchanged", "",
              f"`dense_candidates(depth={DEPTH})` versus exact kNN (exact top {EF} chunks, best chunk "
              f"per RFQ, first {DEPTH}), {s['db_rfqs']} RFQs: same RFQ ids in the same order for "
              f"**{s['identical']} / {s['total']}** queries ({s['n_family']} family + {s['n_code']} code).",
              "", f"Scan node with the fix settings: {s['scan']}.", "",
              f"Result: **{'PASS' if small_passes(s) else 'FAIL'}** (criterion: all {s['total']} identical).", ""]
    if "large" in res:
        g = res["large"]
        L += ["## Check 2: large corpus uses the index", "",
              f"{g['n_queries']} queries (seed {g['query_seed']}), {g['db_rfqs']} RFQs. Reference: exact "
              f"RFQ-level top {DEPTH}. Latency is SQL time after {g['warmup']} warm-up queries, "
              "single client, indicative only.", "",
              "| path | scan | RFQ recall@50 | RFQ recall@10 | mean ms | p50 ms | p95 ms |",
              "|---|---|---|---|---|---|---|"]
        for name in ("before", "after"):
            r = g[name]
            L.append(f"| {name} | {r['scan']} | {r['recall_at_50']:.4f} | {r['recall_at_10']:.4f} | "
                     f"{r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
        L += ["", "`before`: ef_search 200, default planner. `after`: shipped `dense_candidates`.", "",
              "Plan after the change:", "", "```"]
        L += short_plan(g["after"]["plan"]).splitlines()[:3]
        L += ["```", "",
              f"Result: **{'PASS' if large_passes(g) else 'FAIL'}** (criteria: plan after is the HNSW "
              f"index scan; mean recall@50 >= {MIN_RECALL}).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- database
def old_dense(conn, vec, depth: int = DEPTH) -> list[str]:
    """The pre-fix path: ef_search only, default planner. RFQ ids, best chunk first."""
    cand = depth * 4
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(min(cand, 1000))])
        rows = conn.execute(DENSE_CANDIDATE_SQL, {"v": vec, "cand": cand}).fetchall()
        raise psycopg.Rollback()
    return collapse_to_rfq([r[0] for r in rows])[:depth]


def exact_rfq_top(conn, vec, depth: int = DEPTH) -> list[str]:
    rows, _ = exact_topk(conn, vec, depth * 4)
    return collapse_to_rfq([r[1] for r in rows])[:depth]


def plan_node(conn, vec, seqscan_off: bool) -> tuple[str, str]:
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(EF)])
        if seqscan_off:
            conn.execute("SET LOCAL enable_seqscan = off")
        plan = _plan(conn, vec, k=EF)
        raise psycopg.Rollback()
    return scan_node(plan), plan


def run_small(conn, texts: list[str], vecs: list, n_family: int) -> dict:
    emb = PrecomputedEmbedder(dict(zip(texts, vecs)))
    got = [[g[0] for g in dense_candidates(conn, emb, t, DEPTH)] for t in texts]
    truth = [exact_rfq_top(conn, v) for v in vecs]
    node, plan = plan_node(conn, vecs[0], True)
    n_rfq = conn.execute("SELECT count(*) FROM rfqs").fetchone()[0]
    conn.rollback()
    return {"db_rfqs": n_rfq, "total": len(texts), "n_family": n_family,
            "n_code": len(texts) - n_family, "identical": count_identical(got, truth),
            "scan": node, "plan": plan}


def _timed(fn, items: list, warmup: int) -> tuple[list, list[float]]:
    for x in items[:warmup]:
        fn(x)
    out, ms = [], []
    for x in items:
        t0 = time.perf_counter()
        out.append(fn(x))
        ms.append((time.perf_counter() - t0) * 1000)
    return out, ms


def _score(found: list[list[str]], truth: list[list[str]], ms: list[float], scan: str, plan: str) -> dict:
    n = len(truth)
    return {"scan": scan, "plan": plan,
            "recall_at_50": sum(recall_at_k(f, t, 50) for f, t in zip(found, truth)) / n,
            "recall_at_10": sum(recall_at_k(f, t, 10) for f, t in zip(found, truth)) / n,
            **latency_stats(ms)}


def run_large(conn, texts: list[str], vecs: list, warmup: int = WARMUP, log=print) -> dict:
    truth = [exact_rfq_top(conn, v) for v in vecs]
    log("reference done")
    emb = PrecomputedEmbedder(dict(zip(texts, vecs)))
    before, b_ms = _timed(lambda v: old_dense(conn, v), vecs, warmup)
    b_node, b_plan = plan_node(conn, vecs[0], False)
    after, a_ms = _timed(lambda t: [g[0] for g in dense_candidates(conn, emb, t, DEPTH)], texts, warmup)
    a_node, a_plan = plan_node(conn, vecs[0], True)
    n_rfq = conn.execute("SELECT count(*) FROM rfqs").fetchone()[0]
    conn.rollback()
    return {"db_rfqs": n_rfq, "n_queries": len(texts), "query_seed": QUERY_SEED, "warmup": warmup,
            "before": _score(before, truth, b_ms, b_node, b_plan),
            "after": _score(after, truth, a_ms, a_node, a_plan)}


# ---------------------------------------------------------------- commands
def _read_queries(name: str) -> list[str]:
    path = ROOT / "eval" / name
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save(args, key: str, value: dict) -> None:
    out = Path(args.out_dir)
    res = merge_results(read_json(JSON_NAME, out), key, value)
    write_json(JSON_NAME, res, out)
    md = render_md(res)
    (out / "hnsw_fix.md").write_text(md, encoding="utf-8")
    print(md)


def _embed(texts: list[str]) -> list:
    from app.embeddings import E5Embedder

    emb = E5Embedder(get_settings().embedding_model)
    return [emb.embed_query(t) for t in texts]


def cmd_small(args) -> None:
    fam, code = _read_queries("queries.jsonl"), _read_queries("code_queries.jsonl")
    texts = fam + code
    conn = connect(args.db_url)
    _save(args, "small", run_small(conn, texts, _embed(texts), len(fam)))


def cmd_large(args) -> None:
    texts = [q["text"] for q in generate_queries(seed=QUERY_SEED, n=N_QUERIES)]
    conn = connect(args.db_url)
    _save(args, "large", run_large(conn, texts, _embed(texts)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("small")
    p.add_argument("--db-url", default=get_settings().database_url)
    p = sub.add_parser("large")
    p.add_argument("--db-url", default=swap_db_url(get_settings().database_url, SCALE_DB))
    args = ap.parse_args()
    {"small": cmd_small, "large": cmd_large}[args.cmd](args)


if __name__ == "__main__":
    main()
