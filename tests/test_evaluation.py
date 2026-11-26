from app.evaluation import (
    first_relevant_rank, is_cross_language, majority_languages, render_markdown, summarize,
)


def test_majority_language_and_cross():
    maj = majority_languages([("f1", "en"), ("f1", "en"), ("f1", "de"), ("f2", "tr")])
    assert maj == {"f1": "en", "f2": "tr"}
    assert not is_cross_language("en", maj["f1"])
    assert is_cross_language("de", maj["f1"])


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3
    assert first_relevant_rank(["a"], {"z"}) is None


def _row(mode, cross, h5, r10, rr, ms=10.0):
    return {"mode": mode, "cross_language": cross, "hit@5": h5, "p@5": h5 / 2, "recall@10": r10,
            "rr": rr, "latency_ms": ms, "n_relevant": 10}


def test_summarize_splits_and_markdown():
    rows = [_row("dense", False, 1.0, 1.0, 1.0), _row("dense", True, 0.0, 0.5, 0.25, 20.0)]
    s = summarize(rows)["dense"]
    assert s["overall"]["n"] == 2 and s["same_language"]["n"] == 1 and s["cross_language"]["n"] == 1
    assert s["overall"]["recall@10"] == 0.75
    assert s["mean_latency_ms"] == 15.0
    md = render_markdown(summarize(rows))
    assert "| dense | 0.500 | 0.250 | 0.750 | 0.625 |" in md


def test_split_families_deterministic_and_disjoint():
    from app.evaluation import split_families
    fams = {f"fam-{i:03d}" for i in range(40)}
    a = split_families(fams, seed=11)
    b = split_families(fams, seed=11)
    assert a == b
    dev, test = a
    assert not (dev & test)
    assert dev | test == fams
    assert abs(len(dev) - len(test)) <= 1
