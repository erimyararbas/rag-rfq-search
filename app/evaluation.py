"""Helpers for the retrieval evaluation: language split, aggregation, Markdown."""
import random
from collections import Counter, defaultdict

METRICS = ("hit@5", "p@5", "recall@10", "rr")


def split_families(family_ids, seed: int = 11) -> tuple[set[str], set[str]]:
    """Deterministic dev/test split by family: sort, shuffle with Random(seed), first half = dev."""
    fams = sorted(set(family_ids))
    random.Random(seed).shuffle(fams)
    half = (len(fams) + 1) // 2
    return set(fams[:half]), set(fams[half:])


def majority_languages(rows: list[tuple[str, str]]) -> dict[str, str]:
    """(family_id, language) pairs -> majority language per family (ties: alphabetical)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for family, lang in rows:
        counts[family][lang] += 1
    return {
        fam: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for fam, c in counts.items()
    }


def is_cross_language(query_lang: str, family_lang: str) -> bool:
    return query_lang != family_lang


def first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return i
    return None


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _agg(rows: list[dict]) -> dict:
    out = {"n": len(rows), "mean_relevant": _mean([r["n_relevant"] for r in rows])}
    for m in METRICS:
        out[m] = _mean([r[m] for r in rows])
    return out


def summarize(per_query: list[dict]) -> dict:
    """Aggregate per-query rows into {mode: {overall, same_language, cross_language, latency}}."""
    modes = list(dict.fromkeys(r["mode"] for r in per_query))
    summary = {}
    for mode in modes:
        rows = [r for r in per_query if r["mode"] == mode]
        summary[mode] = {
            "overall": _agg(rows),
            "same_language": _agg([r for r in rows if not r["cross_language"]]),
            "cross_language": _agg([r for r in rows if r["cross_language"]]),
            "mean_latency_ms": _mean([r["latency_ms"] for r in rows]),
        }
    return summary


def _table(summary: dict, title: str, key: str) -> list[str]:
    first = next(iter(summary.values()))[key]
    lines = [
        f"{title} (n = {first['n']})",
        "",
        "| mode | hit@5 | P@5 | R@10 | MRR@20 |",
        "|---|---|---|---|---|",
    ]
    for mode, s in summary.items():
        a = s[key]
        lines.append(f"| {mode} | {a['hit@5']:.3f} | {a['p@5']:.3f} | {a['recall@10']:.3f} | {a['rr']:.3f} |")
    return lines


def render_markdown(summary: dict, test_summary: dict | None = None) -> str:
    first = next(iter(summary.values()))["overall"]
    out = [f"Mean relevant RFQs per query: {first['mean_relevant']:.2f}", ""]
    out += _table(summary, "Overall", "overall") + [""]
    out += _table(summary, "Same-language queries", "same_language") + [""]
    out += _table(summary, "Cross-language queries", "cross_language") + [""]
    if test_summary:
        out += _table(test_summary, "Held-out test split (families from split_families, seed 11)", "overall") + [""]
    out += ["Mean query latency per mode (ms): " + ", ".join(
        f"{m} {s['mean_latency_ms']:.1f}" for m, s in summary.items())]
    return "\n".join(out) + "\n"
