"""
faiss_index.py
--------------
DATA STRUCTURE: FAISS Approximate Nearest Neighbor Index
---------------------------------------------------------
Third retrieval method alongside baseline and k-NN graph.

FAISS (Facebook AI Similarity Search) builds an IVF (Inverted File Index):
- Clusters all embeddings into nlist=100 Voronoi cells at build time
- At query time, only searches nprobe=10 nearest cells (not all)
- Much faster than brute-force for large datasets
- Small accuracy tradeoff (approximate, not exact)

This gives us a clean 3-way comparison:
  Baseline  = exact, brute-force, O(n)
  Graph     = approximate, graph traversal, O(log n) amortized
  FAISS IVF = approximate, cluster-based ANN, sub-linear
"""

import numpy as np
import os
import pickle
from typing import List, Tuple, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss-cpu not installed. Run: pip install faiss-cpu")


class FAISSIndex:
    """
    Wraps a FAISS IVFFlat index for approximate nearest neighbor search.
    Falls back to exact search (IndexFlatIP) if dataset is too small for IVF.
    """

    def __init__(self, nlist: int = 100, nprobe: int = 10):
        self.nlist  = nlist
        self.nprobe = nprobe
        self.index  = None
        self.dim    = None
        self.n      = 0

    def build(self, embeddings: np.ndarray):
        """
        Builds the FAISS index from (N, 512) L2-normalized embeddings.
        Uses IVFFlat with inner product (= cosine similarity for L2-normalized vecs).
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu not installed.")

        embeddings = embeddings.astype(np.float32)
        N, D = embeddings.shape
        self.dim = D
        self.n   = N

        if N < self.nlist * 10:
            # Dataset too small for IVF - use exact flat index
            print(f"Dataset size {N} too small for IVF, using exact IndexFlatIP")
            self.index = faiss.IndexFlatIP(D)
        else:
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFFlat(quantizer, D, self.nlist, faiss.METRIC_INNER_PRODUCT)
            print(f"Training FAISS IVF index (nlist={self.nlist})...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe

        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={D}")

    def save(self, path: str):
        if not FAISS_AVAILABLE or self.index is None:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self.index, path)
        print(f"FAISS index saved to {path}")

    def load(self, path: str):
        if not FAISS_AVAILABLE:
            return
        self.index = faiss.read_index(path)
        self.index.nprobe = self.nprobe
        print(f"FAISS index loaded: {self.index.ntotal} vectors")

    def search(
        self,
        query_vec: np.ndarray,
        candidate_indices: Optional[List[int]],
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Searches for top-k nearest neighbors.
        If candidate_indices provided, filters results to those indices only.

        Returns [(product_idx, score), ...] sorted highest first.
        """
        if not FAISS_AVAILABLE or self.index is None:
            raise RuntimeError("FAISS index not built or loaded.")

        query = query_vec.astype(np.float32).reshape(1, -1)

        # Search more than k to allow post-filtering by candidate_indices
        fetch_k = min(k * 10, self.index.ntotal) if candidate_indices else k

        scores, indices = self.index.search(query, fetch_k)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        results = []
        candidate_set = set(candidate_indices) if candidate_indices else None

        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            if candidate_set is not None and idx not in candidate_set:
                continue
            results.append((int(idx), float(score)))
            if len(results) >= k:
                break

        return results
