"""Answer evaluation (costs API tokens). Requires ANTHROPIC_API_KEY and --yes."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

OFF_TOPIC = ["Need a quote for 500 cotton T-shirts", "Wedding catering for 80 guests"]
NO_MATCH_PHRASES = (
    "no reasonable match", "no match", "none of the", "not a reasonable match",
    "no similar", "no relevant", "do not contain", "does not contain",
)


class UsageCounter:
    """Thin proxy around the client that sums usage over messages.create calls."""

    def __init__(self, client):
        self._client = client
        self.input_tokens = 0
        self.output_tokens = 0
        self.messages = self

    def create(self, **kwargs):
        resp = self._client.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return resp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="confirm that API tokens may be spent")
    args = ap.parse_args()
    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set; refusing to run.", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        print("ERROR: this script calls the Claude API and spends tokens; "
              "pass --yes to confirm.", file=sys.stderr)
        sys.exit(1)

    import anthropic

    from app.answer import answer_query
    from app.db import connect
    from app.embeddings import E5Embedder
    from app.retrieval import search

    client = UsageCounter(anthropic.Anthropic(api_key=settings.anthropic_api_key))
    conn = connect()
    embedder = E5Embedder(settings.embedding_model)
    family_of = dict(conn.execute("SELECT rfq_id, family_id FROM rfqs").fetchall())
    lines = (ROOT / "eval" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    queries = [json.loads(line) for line in lines if line.strip()][:20]

    total = valid = family_ok = 0
    for q in queries:
        hits = search(conn, embedder, q["text"], k=5, mode="hybrid")
        res = answer_query(client, settings.claude_model, q["text"], hits)
        sent = {h.rfq_id for h in hits}
        for c in res.citations:
            total += 1
            valid += c.rfq_id in sent
            family_ok += family_of.get(c.rfq_id) == q["family_id"]

    handled = 0
    for text in OFF_TOPIC:
        hits = search(conn, embedder, text, k=5, mode="hybrid")
        res = answer_query(client, settings.claude_model, text, hits)
        handled += (not res.citations) or any(p in res.answer.lower() for p in NO_MATCH_PHRASES)

    def rate(a, b):
        return a / b if b else 0.0

    print(f"citations: {total}")
    print(f"citation validity rate: {rate(valid, total):.3f}")
    print(f"citation family-precision: {rate(family_ok, total):.3f}")
    print(f"off-topic handled: {handled}/{len(OFF_TOPIC)}")
    print(f"tokens: input={client.input_tokens} output={client.output_tokens}")


if __name__ == "__main__":
    main()
