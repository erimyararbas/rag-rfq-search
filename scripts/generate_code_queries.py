"""Write the exact-code query set (seed 23) from data/rfqs.jsonl to eval/code_queries.jsonl."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data.code_queries import SEED, generate_code_queries  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rfqs", default=str(ROOT / "data" / "rfqs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "code_queries.jsonl"))
    args = ap.parse_args()

    rfqs = [json.loads(line) for line in
            Path(args.rfqs).read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = generate_code_queries(rfqs, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"wrote {len(queries)} queries")
    for (t, lang), n in sorted(Counter((q["type"], q["language"]) for q in queries).items()):
        print(f"  {t:9s} {lang:5s} {n}")
    for t in ("single", "pair", "sentence"):
        ns = [q["n_relevant"] for q in queries if q["type"] == t]
        print(f"n_relevant {t}: n={len(ns)} min={min(ns)} mean={sum(ns) / len(ns):.1f} max={max(ns)}")
    ns = [q["n_relevant"] for q in queries]
    print(f"n_relevant all: min={min(ns)} mean={sum(ns) / len(ns):.1f} max={max(ns)}")


if __name__ == "__main__":
    main()
