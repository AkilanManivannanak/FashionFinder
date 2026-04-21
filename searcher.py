"""
engine/searcher.py
-------------------
Unified search interface exposing two retrieval methods:

1. BASELINE  - brute-force cosine similarity over the full dataset
               (or hash-filtered category subset)
2. GRAPH     - k-NN graph traversal with heap-ranked results

Both return the same output format so the API and UI can
switch between them with a single flag and compare results.
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional

from hash_index import HashIndex
from heap_ranker import top_k_cosine
from knn_graph import KNNGraph


class SearchResult:
    """Single result item returned by both search methods."""
    def __init__(self, rank: int, product_idx: int, score: float, metadata: dict):
        self.rank = rank
        self.product_idx = product_idx
        self.score = score
        self.metadata = metadata  # dict with id, name, category, etc.

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "product_idx": self.product_idx,
            "score": round(self.score, 4),
            **self.metadata
        }


class Searcher:
    """
    Wraps both retrieval strategies behind a single .search() call.
    Measures and returns latency for every query so we can benchmark
    baseline vs graph side by side.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        hash_index: HashIndex,
        knn_graph: Optional[KNNGraph] = None,
        images_dir: str = "data/fashion-dataset/images"
    ):
        self.embeddings = embeddings          # (N, 512)
        self.metadata = metadata              # DataFrame
        self.hash_index = hash_index
        self.knn_graph = knn_graph
        self.images_dir = images_dir
        print(f"Searcher ready: {len(embeddings)} products indexed")

    def _row_to_meta(self, idx: int) -> dict:
        """Converts a metadata row to a clean dict for API output."""
        row = self.metadata.loc[idx]
        product_id = int(row.get("id", idx))
        return {
            "id": product_id,
            "name": str(row.get("productDisplayName", "Unknown")),
            "masterCategory": str(row.get("masterCategory", "Unknown")),
            "subCategory": str(row.get("subCategory", "Unknown")),
            "articleType": str(row.get("articleType", "Unknown")),
            "baseColour": str(row.get("baseColour", "Unknown")),
            "season": str(row.get("season", "Unknown")),
            "year": str(row.get("year", "Unknown")),
            "image_path": f"{self.images_dir}/{product_id}.jpg"
        }

    def search_baseline(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        category: Optional[str] = None,
        query_idx: Optional[int] = None
    ) -> Tuple[List[SearchResult], float]:
        """
        BASELINE: Brute-force cosine similarity.
        Optionally filtered to a category bucket via hash index.

        Returns: (results, latency_ms)
        """
        t0 = time.perf_counter()

        # Get candidate indices from hash table
        if category:
            candidate_indices = self.hash_index.get_indices(category)
        else:
            candidate_indices = list(range(len(self.embeddings)))

        # Exclude the query itself if it is in the dataset
        if query_idx is not None and query_idx in candidate_indices:
            candidate_indices = [i for i in candidate_indices if i != query_idx]

        # Heap-ranked top-k cosine similarity
        top_k = top_k_cosine(query_vec, self.embeddings, candidate_indices, k)

        latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            SearchResult(rank=i + 1, product_idx=idx, score=score,
                         metadata=self._row_to_meta(idx))
            for i, (idx, score) in enumerate(top_k)
        ]

        return results, latency_ms

    def search_graph(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        category: Optional[str] = None,
        query_idx: Optional[int] = None,
        expansion_hops: int = 2
    ) -> Tuple[List[SearchResult], float]:
        """
        GRAPH: k-NN graph traversal with heap-ranked results.
        Entry point comes from hash-filtered category candidates.

        Returns: (results, latency_ms)
        """
        if self.knn_graph is None:
            raise RuntimeError("k-NN graph not loaded. Call with method=baseline.")

        t0 = time.perf_counter()

        # Use hash index to get entry candidates
        if category:
            entry_candidates = self.hash_index.get_indices(category)
        else:
            # Sample entry points if no category filter (avoid full scan)
            all_idx = list(range(len(self.embeddings)))
            entry_candidates = all_idx[:500]  # seed with first 500

        if query_idx is not None:
            entry_candidates = [i for i in entry_candidates if i != query_idx]

        top_k = self.knn_graph.search(
            query_vec, self.embeddings, entry_candidates, k, expansion_hops
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            SearchResult(rank=i + 1, product_idx=idx, score=score,
                         metadata=self._row_to_meta(idx))
            for i, (idx, score) in enumerate(top_k)
        ]

        return results, latency_ms

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        method: str = "graph",
        category: Optional[str] = None,
        query_idx: Optional[int] = None
    ) -> Dict:
        """
        Main entry point. Runs the selected method and returns
        results + latency + method metadata for API/UI consumption.

        method: "baseline" or "graph"
        """
        if method == "baseline":
            results, latency_ms = self.search_baseline(
                query_vec, k, category, query_idx)
        elif method == "graph":
            results, latency_ms = self.search_graph(
                query_vec, k, category, query_idx)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'baseline' or 'graph'.")

        return {
            "method": method,
            "latency_ms": round(latency_ms, 2),
            "k": k,
            "category_filter": category,
            "results": [r.to_dict() for r in results]
        }
