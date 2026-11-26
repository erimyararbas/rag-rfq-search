from app.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 2) == 1 / 3


def test_recall_empty():
    assert recall_at_k([], {"a"}, 5) == 0.0


def test_mrr():
    assert mrr(["x", "a"], {"a"}) == 0.5
    assert mrr(["x"], {"a"}) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_rate_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_precision():
    assert precision_at_k(["a", "x", "c", "y"], {"a", "c"}, 4) == 0.5
    assert precision_at_k(["a"], {"a"}, 5) == 0.2


def test_recall_empty_relevant():
    assert recall_at_k(["a"], set(), 5) == 0.0
