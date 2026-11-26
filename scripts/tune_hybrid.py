"""Pre-registered weighted-RRF experiment: tune w_lex and k on dev, report on held-out test."""
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.db import connect  # noqa: E402
from app.embeddings import E5Embedder  # noqa: E402
from app.evaluation import _agg, split_families  # noqa: E402
from app.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k  # noqa: E402
from app.retrieval import DEPTH, dense_candidates, lexical_candidates, rrf  # noqa: E402

SPLIT_SEED = 11
W_LEX_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
K_GRID = (10, 60)
K_RETRIEVED = 20

PROTOCOL = """\
Protocol (split and grid defined after a first full-set run; fixed before the tuning run and not changed after seeing its results)
- Split: by family. Sorted unique family_ids of eval/queries.jsonl, shuffled with random.Random(11); the first half is dev, the rest is test. Both queries of a family land in the same split.
- Fusion: weighted RRF, score(d) = w_dense/(k + rank_dense(d)) + w_lex/(k + rank_lex(d)), with w_dense = 1.0. Candidate depth per list: 50.
- Grid: w_lex in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0} x k in {10, 60} (12 configs).
- Selection metric: dev MRR@20. Tie-break: higher dev hit@5, then smaller w_lex, then k = 60.
- Compared on dev and test: dense, lexical, hybrid (w_lex = 1, k = 60, the previous default) and hybrid (selected config). Test numbers are reported whatever they show. No second selection round.
- The selected config becomes the default in Settings, even if its w_lex is 0.0 (which would mean lexical adds nothing on this data).
- The split and grid were defined after a first full-set run had shown equal-weight hybrid underperforming. Test families were held out from the selection step, not from all observation.
"""


def git_commit():
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def score_row(qid: str, family: str, ranked: list[str], relevant: set[str]) -> dict:
    return {
        "query_id": qid, "family_id": family, "n_relevant": len(relevant),
        "hit@5": hit_rate_at_k(ranked, relevant, 5),
        "p@5": precision_at_k(ranked, relevant, 5),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "rr": mrr(ranked, relevant),
    }


def config_name(w: float, k: int) -> str:
    return f"w_lex={w}, k={k}"


def selection_key(r: dict):
    # max(): higher MRR@20, higher hit@5, smaller w_lex, then k = 60
    return (r["dev"]["rr"], r["dev"]["hit@5"], -r["w_lex"], 1 if r["k"] == 60 else 0)


def table(title: str, systems: list[tuple[str, dict]]) -> list[str]:
    lines = [title, "", "| system | hit@5 | P@5 | R@10 | MRR@20 |", "|---|---|---|---|---|"]
    for name, a in systems:
        lines.append(f"| {name} | {a['hit@5']:.3f} | {a['p@5']:.3f} | {a['recall@10']:.3f} | {a['rr']:.3f} |")
    return lines


def main() -> None:
    queries = [json.loads(line) for line in
               (ROOT / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    conn = connect()
    family_rfqs: dict[str, set[str]] = {}
    for fam, rid in conn.execute("SELECT family_id, rfq_id FROM rfqs").fetchall():
        family_rfqs.setdefault(fam, set()).add(rid)
    dev_fams, test_fams = split_families({q["family_id"] for q in queries}, seed=SPLIT_SEED)

    def split_of(q):
        return "dev" if q["family_id"] in dev_fams else "test"

    embedder = E5Embedder(get_settings().embedding_model)
    # Candidate lists once per query; all fusion happens in Python.
    cands = {}
    for q in queries:
        dense = [r[0] for r in dense_candidates(conn, embedder, q["text"], DEPTH)]
        lex = [r[0] for r in lexical_candidates(conn, q["text"], DEPTH)]
        cands[q["query_id"]] = (dense, lex)

    def system_rows(rank_fn) -> list[dict]:
        out = []
        for q in queries:
            dense, lex = cands[q["query_id"]]
            ranked = rank_fn(dense, lex)[:K_RETRIEVED]
            row = score_row(q["query_id"], q["family_id"], ranked, family_rfqs[q["family_id"]])
            row["split"] = split_of(q)
            out.append(row)
        return out

    def hybrid_fn(w, k):
        return lambda d, lx: [i for i, _ in rrf([d, lx], k=k, weights=[1.0, w])]

    def agg_split(rows, split):
        return _agg([r for r in rows if r["split"] == split])

    grid = []
    for w, k in itertools.product(W_LEX_GRID, K_GRID):
        rows = system_rows(hybrid_fn(w, k))
        grid.append({"w_lex": w, "k": k, "dev": agg_split(rows, "dev"), "rows": rows})
    best = max(grid, key=selection_key)
    sel_name = config_name(best["w_lex"], best["k"])

    systems = {
        "dense": system_rows(lambda d, lx: d),
        "lexical": system_rows(lambda d, lx: lx),
        "hybrid (w_lex=1.0, k=60, previous default)": next(
            g["rows"] for g in grid if g["w_lex"] == 1.0 and g["k"] == 60),
        f"hybrid selected ({sel_name})": best["rows"],
    }
    comparison = {
        name: {"dev": agg_split(rows, "dev"), "test": agg_split(rows, "test")}
        for name, rows in systems.items()
    }

    n_dev = sum(1 for q in queries if split_of(q) == "dev")
    grid_lines = ["| w_lex | k | hit@5 | P@5 | R@10 | MRR@20 |", "|---|---|---|---|---|---|"]
    for g in grid:
        a = g["dev"]
        grid_lines.append(
            f"| {g['w_lex']} | {g['k']} | {a['hit@5']:.3f} | {a['p@5']:.3f} | {a['recall@10']:.3f} | {a['rr']:.3f} |")
    md = "\n".join(
        [PROTOCOL, f"Dev: {len(dev_fams)} families, {n_dev} queries. Test: {len(test_fams)} families, "
                   f"{len(queries) - n_dev} queries.", "",
         "Grid on dev (12 configs)", ""] + grid_lines + [
         "", f"Selected: w_lex = {best['w_lex']}, k = {best['k']} (dev MRR@20 {best['dev']['rr']:.3f}).", ""]
        + table("Dev split (for transparency)", [(n, c["dev"]) for n, c in comparison.items()]) + [""]
        + table("Test split (held out)", [(n, c["test"]) for n, c in comparison.items()]) + [""]
        + (["Selected w_lex is 0.0: lexical adds nothing on this data.", ""] if best["w_lex"] == 0.0 else [])
    )

    result = {
        "protocol": PROTOCOL, "split_seed": SPLIT_SEED,
        "dev_families": sorted(dev_fams), "test_families": sorted(test_fams),
        "grid": {"w_lex": list(W_LEX_GRID), "k": list(K_GRID)},
        "dev_grid_results": [{"w_lex": g["w_lex"], "k": g["k"], **g["dev"]} for g in grid],
        "selected": {"w_lex": best["w_lex"], "k": best["k"]},
        "comparison": comparison,
        "per_query": [{"system": name, **r} for name, rows in systems.items() for r in rows],
        "embedding_model": get_settings().embedding_model,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hybrid_tuning.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "hybrid_tuning.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
