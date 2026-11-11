import numpy as np
import pytest

from mmr_reranker import MMRReranker, apply_mmr


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def near_duplicates():
    """Query plus 6 items: 4 near-copies of one look, 2 different looks that are a bit less relevant."""
    q = _unit([1, 0, 0, 0])
    base = _unit([0.95, 0.31, 0, 0])
    emb = np.stack([
        base, _unit(base + [0, 0.01, 0, 0]), _unit(base + [0, 0.02, 0, 0]), _unit(base + [0, 0.03, 0, 0]),
        _unit([0.85, 0, 0.53, 0]), _unit([0.80, 0, 0, 0.60]),
    ])
    results = sorted(((i, float(emb[i] @ q)) for i in range(len(emb))), key=lambda t: -t[1])
    return q, emb, results


def _mean_pairwise_distance(emb, idx):
    sub = emb[idx]
    sims = sub @ sub.T
    n = len(idx)
    return float((1 - sims)[~np.eye(n, dtype=bool)].mean())


def test_lambda_one_keeps_relevance_order(near_duplicates):
    q, emb, results = near_duplicates
    out = apply_mmr(q, results, emb, k=6, lambda_param=1.0)
    assert out == results


def test_lower_lambda_increases_diversity(near_duplicates):
    q, emb, results = near_duplicates
    k = 3
    pure = [i for i, _ in apply_mmr(q, results, emb, k=k, lambda_param=1.0)]
    diverse = [i for i, _ in apply_mmr(q, results, emb, k=k, lambda_param=0.3)]
    assert _mean_pairwise_distance(emb, diverse) > _mean_pairwise_distance(emb, pure)
    assert {4, 5} & set(diverse), "a different look should enter the top 3 when diversity is weighted"


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.6, 1.0])
@pytest.mark.parametrize("k", [1, 3, 6, 10])
def test_output_is_unique_and_sized(near_duplicates, lam, k):
    q, emb, results = near_duplicates
    out = apply_mmr(q, results, emb, k=k, lambda_param=lam)
    ids = [i for i, _ in out]
    assert len(ids) == min(k, len(results)) == len(set(ids))
    assert dict(out).items() <= dict(results).items(), "original scores are preserved"


def test_first_pick_is_most_relevant(near_duplicates):
    q, emb, results = near_duplicates
    for lam in (0.2, 0.6, 1.0):
        assert apply_mmr(q, results, emb, k=3, lambda_param=lam)[0][0] == results[0][0]


def test_empty_input():
    assert MMRReranker().rerank(np.zeros(4), [], [], np.zeros((0, 4))) == []
