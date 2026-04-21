"""
engine/knn_graph.py
--------------------
DATA STRUCTURE: k-NN Graph stored as Adjacency List
-----------------------------------------------------
Each product is a NODE.
Each node has edges to its k most similar neighbors,
computed once from precomputed embeddings and stored as:

    adjacency_list: Dict[int, List[Tuple[int, float]]]
    {
        product_idx: [(neighbor_idx, similarity_score), ...],
        ...
    }

At query time, instead of scanning ALL products, we:
1. Find the entry point (nearest node to the query using hash-filtered brute force)
2. Traverse the graph greedily, following edges to neighbors
3. Use a HeapRanker to track the top-k best results seen

This is much faster than brute force as the dataset grows.
Build once. Query many times.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
import os

from heap_ranker import HeapRanker, top_k_cosine


class KNNGraph:
    """
    k-NN similarity graph stored as an adjacency list.
    Built once from precomputed embeddings, saved to disk,
    loaded for fast query-time traversal.
    """

    def __init__(self, k_neighbors: int = 10):
        """
        k_neighbors: how many edges each node gets.
        Higher k = better recall, slower build, more memory.
        k=10 is a good default for this dataset size.
        """
        self.k_neighbors = k_neighbors
        # adjacency_list[i] = [(neighbor_idx, score), ...]
        self.adjacency_list: Dict[int, List[Tuple[int, float]]] = {}
        self.n_nodes = 0

    def build(self, embeddings: np.ndarray, batch_size: int = 500):
        """
        Builds the k-NN graph from an (N, 512) embedding matrix.
        For each product, finds its k nearest neighbors by cosine similarity.

        Uses batched matrix multiplication for speed.
        Time: O(N^2 / batch_size) -- manageable for 10K-50K products.
        """
        N = len(embeddings)
        self.n_nodes = N
        print(f"Building k-NN graph: {N} nodes, k={self.k_neighbors} neighbors each")

        for start in tqdm(range(0, N, batch_size), desc="Building graph"):
            end = min(start + batch_size, N)
            batch = embeddings[start:end]           # (batch, 512)
            sims = batch @ embeddings.T             # (batch, N) cosine similarities

            for local_i, global_i in enumerate(range(start, end)):
                row = sims[local_i]                 # (N,) similarity to all others
                row[global_i] = -1.0                # exclude self

                # Get top-k neighbor indices (unsorted first, then sort)
                top_k_idx = np.argpartition(row, -self.k_neighbors)[-self.k_neighbors:]
                top_k_sorted = top_k_idx[np.argsort(row[top_k_idx])[::-1]]

                self.adjacency_list[global_i] = [
                    (int(j), float(row[j])) for j in top_k_sorted
                ]

        print(f"Graph built: {len(self.adjacency_list)} nodes, "
              f"{sum(len(v) for v in self.adjacency_list.values())} total edges")

    def save(self, path: str):
        """Save the adjacency list to disk with pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "adjacency_list": self.adjacency_list,
                "k_neighbors": self.k_neighbors,
                "n_nodes": self.n_nodes
            }, f)
        print(f"Graph saved to {path}")

    def load(self, path: str):
        """Load adjacency list from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.adjacency_list = data["adjacency_list"]
        self.k_neighbors = data["k_neighbors"]
        self.n_nodes = data["n_nodes"]
        print(f"Graph loaded: {self.n_nodes} nodes, k={self.k_neighbors}")

    def search(
        self,
        query_vec: np.ndarray,
        embeddings: np.ndarray,
        entry_candidates: List[int],
        k: int = 10,
        expansion_hops: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Graph traversal search.

        Steps:
        1. Find the best entry point from entry_candidates
           (these come from the hash index category filter, already small)
        2. From that entry point, follow edges (BFS-style expansion)
           for expansion_hops levels
        3. Use HeapRanker to track top-k throughout

        Args:
            query_vec:        (512,) normalized query embedding
            embeddings:       (N, 512) full matrix (for scoring visited nodes)
            entry_candidates: indices pre-filtered by hash index
            k:                number of results
            expansion_hops:   how many graph hops to explore beyond entry

        Returns:
            [(product_idx, score), ...] sorted highest first
        """
        if not self.adjacency_list:
            raise RuntimeError("Graph not built or loaded yet.")

        ranker = HeapRanker(k=k)
        visited = set()

        # ── Step 1: Score entry candidates, seed ranker ──────────────────────
        if len(entry_candidates) > 0:
            cands = np.array(entry_candidates)
            scores = embeddings[cands] @ query_vec
            ranker.push_batch(scores, entry_candidates)
            for idx in entry_candidates:
                visited.add(idx)

        # ── Step 2: Expand via graph edges ───────────────────────────────────
        frontier = [idx for idx, _ in ranker.top_k()]

        for _ in range(expansion_hops):
            next_frontier = []
            for node_idx in frontier:
                neighbors = self.adjacency_list.get(node_idx, [])
                unvisited = [(n_idx, n_score) for n_idx, n_score in neighbors
                             if n_idx not in visited]
                for n_idx, _ in unvisited:
                    visited.add(n_idx)
                    next_frontier.append(n_idx)

            if not next_frontier:
                break

            # Score all new nodes
            nf_arr = np.array(next_frontier)
            scores = embeddings[nf_arr] @ query_vec
            ranker.push_batch(scores, next_frontier)
            frontier = next_frontier

        return ranker.top_k()

    def neighbors(self, idx: int) -> List[Tuple[int, float]]:
        """Returns the precomputed neighbors of a node (for display/debug)."""
        return self.adjacency_list.get(idx, [])
