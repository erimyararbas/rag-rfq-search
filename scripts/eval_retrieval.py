"""Retrieval evaluation: hit@5, P@5, R@10 and MRR@20 for dense, lexical and hybrid modes."""
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
    first_relevant_rank, is_cross_language, majority_languages, render_markdown,
    split_families, summarize,
)
from app.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k  # noqa: E402
from app.retrieval import search  # noqa: E402

MODES = ("dense", "lexical", "hybrid")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None)
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    text = Path(args.queries).read_text(encoding="utf-8")
    queries = [json.loads(line) for line in text.splitlines() if line.strip()]
    conn = connect()
    fam_rows = conn.execute("SELECT family_id, language, rfq_id FROM rfqs").fetchall()
    family_rfqs = {}
    for fam, _, rid in fam_rows:
        family_rfqs.setdefault(fam, set()).add(rid)
    fam_lang = majority_languages([(f, lang) for f, lang, _ in fam_rows])

    settings = get_settings()
    model = settings.embedding_model
    _, test_fams = split_families({q["family_id"] for q in queries}, seed=11)
    embedder = E5Embedder(model)
    per_query = []
    for mode in MODES:
        for q in queries:
            relevant = family_rfqs[q["family_id"]]
            t0 = time.perf_counter()
            hits = search(
                conn, embedder, q["text"], k=20, mode=mode,
                lexical_weight=settings.hybrid_lexical_weight, rrf_k=settings.rrf_k,
            )
            latency = (time.perf_counter() - t0) * 1000
            ranked = [h.rfq_id for h in hits]
            per_query.append({
                "query_id": q["query_id"], "mode": mode, "language": q["language"],
                "family_id": q["family_id"],
                "cross_language": is_cross_language(q["language"], fam_lang[q["family_id"]]),
                "n_relevant": len(relevant),
                "hit@5": hit_rate_at_k(ranked, relevant, 5),
                "p@5": precision_at_k(ranked, relevant, 5),
                "recall@10": recall_at_k(ranked, relevant, 10),
                "rr": mrr(ranked, relevant),
                "first_relevant_rank": first_relevant_rank(ranked, relevant),
                "latency_ms": latency,
            })

    summary = summarize(per_query)
    test_summary = summarize([r for r in per_query if r["family_id"] in test_fams])
    result = {
        "embedding_model": model,
        "corpus_size": conn.execute("SELECT count(*) FROM rfqs").fetchone()[0],
        "query_count": len(queries),
        "mean_relevant_per_query": sum(len(family_rfqs[q["family_id"]]) for q in queries) / len(queries),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(args.commit),
        "k_retrieved": 20,
        "hybrid_lexical_weight": settings.hybrid_lexical_weight,
        "rrf_k": settings.rrf_k,
        "summary": summary,
        "test_summary": test_summary,
        "per_query": per_query,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render_markdown(summary, test_summary)
    (out / "retrieval.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
