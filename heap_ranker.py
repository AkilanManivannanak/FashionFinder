"""
engine/heap_ranker.py
----------------------
DATA STRUCTURE: Min-Heap / Priority Queue
------------------------------------------
Maintains the top-k most similar products during search
without sorting the entire candidate list.

Time complexity: O(n log k) vs O(n log n) for full sort.
For n=10,000 and k=10, this is ~4x fewer operations.

The heap stores tuples of (similarity_score, product_index).
It is a MIN-heap, so the least similar item sits at the top
and gets evicted when a better one arrives, always keeping
exactly the top-k best results.
"""

import heapq
import numpy as np
from typing import List, Tuple


class HeapRanker:
    """
    Min-heap that tracks the top-k highest similarity scores
    seen so far during a scan or graph traversal.
    """

    def __init__(self, k: int):
        self.k = k
        self._heap: List[Tuple[float, int]] = []  # (score, idx) min-heap

    def push(self, score: float, idx: int):
        """
        Add a (score, idx) pair.
        If heap has fewer than k items, always add.
        Otherwise, only add if score beats the current worst.
        """
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (score, idx))
        elif score > self._heap[0][0]:
            # New item beats the worst in heap: evict worst, add new
            heapq.heapreplace(self._heap, (score, idx))

    def push_batch(self, scores: np.ndarray, indices: List[int]):
        """
        Efficiently push a batch of (score, idx) pairs.
        Used after computing cosine similarities over a candidate set.
        """
        for score, idx in zip(scores.tolist(), indices):
            self.push(score, idx)

    def top_k(self) -> List[Tuple[int, float]]:
        """
        Returns the top-k results as [(idx, score), ...] sorted
        highest similarity first.
        """
        # heapq gives smallest first; we want largest first
        return [(idx, score) for score, idx in sorted(self._heap, reverse=True)]

    def reset(self):
        """Clear the heap for reuse."""
        self._heap = []

    def __len__(self):
        return len(self._heap)


def top_k_cosine(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    candidate_indices: List[int],
    k: int
) -> List[Tuple[int, float]]:
    """
    Standalone function: computes cosine similarity between a query vector
    and a subset of embeddings (defined by candidate_indices),
    then uses a HeapRanker to return the top-k results.

    Args:
        query_vec:         (512,) normalized query embedding
        embeddings:        (N, 512) full embedding matrix
        candidate_indices: list of row indices to search within
        k:                 number of results to return

    Returns:
        [(product_idx, similarity_score), ...] sorted highest first
    """
    if len(candidate_indices) == 0:
        return []

    candidates = embeddings[candidate_indices]           # (M, 512)
    scores = candidates @ query_vec                      # (M,) dot = cosine (L2 normalized)

    ranker = HeapRanker(k=k)
    ranker.push_batch(scores, candidate_indices)

    return ranker.top_k()
