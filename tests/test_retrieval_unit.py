from app.retrieval import rrf


def test_rrf_basic():
    assert rrf([["a", "b"], ["b", "c"]])[0][0] == "b"


def test_rrf_score_value():
    scores = dict(rrf([["a", "b"], ["b", "c"]]))
    assert scores["b"] == 1 / 61 + 1 / 62


def test_rrf_empty_list():
    assert rrf([[], ["a"]]) == [("a", 1 / 61)]


def test_weighted_rrf():
    res = rrf([["a", "b"], ["b", "a"]], k=60, weights=[1.0, 0.0])
    assert [r[0] for r in res] == ["a", "b"]
    assert dict(res)["a"] == 1 / 61


def test_rrf_default_weights_unchanged():
    assert rrf([["a", "b"], ["b", "c"]]) == rrf([["a", "b"], ["b", "c"]], weights=[1.0, 1.0])
