import numpy as np
import pytest

from heap_ranker import HeapRanker, top_k_cosine


def _reference(query, emb, cands, k):
    scores = emb[cands] @ query
    order = np.argsort(-scores, kind="stable")[:k]
    return [cands[i] for i in order], scores[order]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [1, 5, 10, 50])
def test_top_k_cosine_matches_numpy_argsort(seed, k):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(200, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[0]
    cands = sorted(rng.choice(200, size=120, replace=False).tolist())
    got = top_k_cosine(q, emb, cands, k)
    ref_idx, ref_scores = _reference(q, emb, cands, k)
    assert len(got) == min(k, len(cands))
    np.testing.assert_allclose([s for _, s in got], ref_scores, rtol=1e-6)
    assert set(i for i, _ in got) == set(ref_idx)


def test_results_are_sorted_descending_and_unique():
    r = HeapRanker(k=4)
    r.push_batch(np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.8]), [10, 11, 12, 13, 14, 15])
    out = r.top_k()
    assert [i for i, _ in out] == [11, 15, 13, 12]
    assert len({i for i, _ in out}) == len(out) == len(r) == 4


def test_k_larger_than_candidates_returns_all():
    emb = np.eye(3, dtype=np.float32)
    assert len(top_k_cosine(emb[0], emb, [0, 1, 2], k=10)) == 3


def test_empty_candidates_and_zero_k():
    emb = np.eye(3, dtype=np.float32)
    assert top_k_cosine(emb[0], emb, [], k=5) == []
    assert top_k_cosine(emb[0], emb, [0, 1, 2], k=0) == []
    r = HeapRanker(k=0)
    r.push(0.5, 1)
    assert r.top_k() == []


def test_ties_keep_k_items():
    r = HeapRanker(k=2)
    for i in range(5):
        r.push(0.5, i)
    assert len(r.top_k()) == 2 and all(s == 0.5 for _, s in r.top_k())


def test_reset_clears_state():
    r = HeapRanker(k=2)
    r.push(1.0, 1)
    r.reset()
    assert len(r) == 0 and r.top_k() == []
