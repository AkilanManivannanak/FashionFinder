"""
mmr_reranker.py
---------------
Maximal Marginal Relevance (MMR) Diversity Reranking
-----------------------------------------------------
Solves the "10 identical blue jerseys" problem.

Standard top-k retrieval returns the most similar items — but they are
often near-duplicates. MMR reranks results to balance:
    - Relevance:  how similar is this item to the query?
    - Diversity:  how different is this item from already-selected results?

Formula:
    MMR(item) = lambda * similarity(item, query)
              - (1 - lambda) * max(similarity(item, selected))

    lambda = 1.0 → pure relevance (same as original ranking)
    lambda = 0.5 → balanced relevance + diversity
    lambda = 0.0 → pure diversity

Time: O(k^2) — fast for k=10 or k=20.
"""

import numpy as np
from typing import List, Tuple


class MMRReranker:
    """
    Maximal Marginal Relevance reranker.
    Takes top-k results and returns a reordered list that balances
    similarity to query with diversity among selected items.
    """

    def __init__(self, lambda_param: float = 0.6):
        """
        lambda_param: trade-off between relevance and diversity.
            0.6 = slightly relevance-biased (good default for fashion search)
            0.5 = perfectly balanced
            0.8 = mostly relevance, some diversity
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        query_vec: np.ndarray,
        candidate_indices: List[int],
        candidate_scores: List[float],
        embeddings: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float, float]]:
        """
        Reranks candidates using MMR.

        Args:
            query_vec:         (512,) normalized query embedding
            candidate_indices: list of candidate product indices
            candidate_scores:  list of similarity scores (same order)
            embeddings:        (N, 512) full embedding matrix
            k:                 number of results to return

        Returns:
            [(product_idx, original_score, mmr_score), ...] MMR-reranked
        """
        if len(candidate_indices) == 0:
            return []

        k = min(k, len(candidate_indices))

        # Build candidate embedding matrix
        cand_embs = embeddings[candidate_indices]  # (M, 512)
        score_map  = {idx: score for idx, score in zip(candidate_indices, candidate_scores)}

        selected   = []       # Final MMR-selected indices
        selected_embs = []    # Embeddings of selected items
        remaining  = list(range(len(candidate_indices)))  # Indices into candidate list

        while len(selected) < k and remaining:
            best_mmr_score = -np.inf
            best_pos       = None

            for pos in remaining:
                rel_score = candidate_scores[pos]   # similarity to query

                if not selected_embs:
                    # Nothing selected yet — pure relevance
                    redundancy = 0.0
                else:
                    # Max similarity to any already-selected item
                    sel_matrix = np.array(selected_embs)   # (s, 512)
                    sims = cand_embs[pos] @ sel_matrix.T   # (s,)
                    redundancy = float(np.max(sims))

                mmr_score = (self.lambda_param * rel_score
                             - (1 - self.lambda_param) * redundancy)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_pos       = pos

            if best_pos is None:
                break

            selected.append((
                candidate_indices[best_pos],
                candidate_scores[best_pos],
                round(best_mmr_score, 4)
            ))
            selected_embs.append(cand_embs[best_pos])
            remaining.remove(best_pos)

        return selected


def apply_mmr(
    query_vec: np.ndarray,
    results: List[Tuple[int, float]],
    embeddings: np.ndarray,
    k: int = 10,
    lambda_param: float = 0.6
) -> List[Tuple[int, float]]:
    """
    Convenience wrapper: takes [(idx, score), ...] and returns
    MMR-reranked [(idx, score), ...] preserving original score.
    """
    reranker = MMRReranker(lambda_param=lambda_param)
    indices  = [r[0] for r in results]
    scores   = [r[1] for r in results]

    mmr_results = reranker.rerank(query_vec, indices, scores, embeddings, k)
    # Return (idx, original_score) preserving original scores
    return [(idx, orig_score) for idx, orig_score, _ in mmr_results]
