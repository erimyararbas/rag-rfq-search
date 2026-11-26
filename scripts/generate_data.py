"""Write the synthetic corpus (seed 42) and eval queries (seed 7) as JSONL."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data.generate import generate_queries, generate_rfqs  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="corpus seed")
    ap.add_argument("--query-seed", type=int, default=7, help="eval query seed")
    args = ap.parse_args()

    rfqs = generate_rfqs(args.seed)
    queries = generate_queries(args.query_seed)
    write_jsonl(ROOT / "data" / "rfqs.jsonl", rfqs)
    write_jsonl(ROOT / "eval" / "queries.jsonl", queries)
    print(f"wrote {len(rfqs)} RFQs and {len(queries)} queries")


if __name__ == "__main__":
    main()
