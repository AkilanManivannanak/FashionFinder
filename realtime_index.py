"""
realtime_index.py
-----------------
Real-Time Index: Add new products instantly without rebuilding FAISS.

Uses a two-tier approach:
  1. Main FAISS index (existing 44,419 products) - fast ANN search
  2. Live buffer (new products added at runtime) - exact brute-force

When searching:
  - Search the FAISS index for top-k
  - Search the live buffer for top-k
  - Merge and re-rank results

New products are immediately searchable without any rebuild.
This is what production systems at Amazon scale use.
"""

import numpy as np
import pandas as pd
import time
import os
import json
import pickle
from typing import List, Tuple, Optional, Dict
from PIL import Image
from heap_ranker import HeapRanker


class RealTimeIndex:
    """
    Wraps the existing FAISS index with a live buffer for real-time updates.
    New products added via add_product() are searchable immediately.
    """

    def __init__(self, embedder, faiss_index, metadata: pd.DataFrame, images_dir: str = "archive-2/images"):
        self.embedder    = embedder
        self.faiss_index = faiss_index
        self.metadata    = metadata
        self.images_dir  = images_dir

        # Live buffer: list of (embedding, metadata_dict)
        self.buffer_embeddings: List[np.ndarray] = []
        self.buffer_metadata:   List[dict]       = []
        self.next_id = int(metadata["id"].max()) + 1 if len(metadata) > 0 else 100000

        # Persist buffer across restarts
        self.buffer_path = "data/realtime_buffer.pkl"
        self._load_buffer()

    def _load_buffer(self):
        """Loads persisted buffer from disk."""
        if os.path.exists(self.buffer_path):
            try:
                with open(self.buffer_path, "rb") as f:
                    data = pickle.load(f)
                self.buffer_embeddings = data.get("embeddings", [])
                self.buffer_metadata   = data.get("metadata", [])
                self.next_id           = data.get("next_id", self.next_id)
                print(f"RealTimeIndex: loaded {len(self.buffer_embeddings)} buffered products")
            except:
                pass

    def _save_buffer(self):
        """Persists buffer to disk."""
        os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)
        with open(self.buffer_path, "wb") as f:
            pickle.dump({
                "embeddings": self.buffer_embeddings,
                "metadata":   self.buffer_metadata,
                "next_id":    self.next_id,
            }, f)

    def add_product(
        self,
        pil_image: Image.Image,
        name: str,
        category: str = "Apparel",
        article_type: str = "Tshirts",
        color: str = "Unknown",
        brand: str = "Unknown",
        season: str = "Summer",
    ) -> dict:
        """
        Adds a new product to the real-time index.
        The product is immediately searchable.

        Returns the new product's metadata dict.
        """
        # Embed the image
        embedding = self.embedder.embed_pil(pil_image)
        if embedding is None:
            raise ValueError("Could not embed image.")

        # Assign a new product ID
        new_id = self.next_id
        self.next_id += 1

        # Save image to disk
        img_path = os.path.join(self.images_dir, f"{new_id}.jpg")
        try:
            pil_image.convert("RGB").save(img_path, "JPEG", quality=85)
        except:
            pass

        # Create metadata
        meta = {
            "id":                 new_id,
            "productDisplayName": name,
            "masterCategory":     category,
            "subCategory":        article_type,
            "articleType":        article_type,
            "baseColour":         color,
            "season":             season,
            "year":               "2026",
            "brand":              brand,
            "image_path":         img_path,
            "added_at":           time.time(),
            "is_realtime":        True,
        }

        # Add to buffer
        self.buffer_embeddings.append(embedding)
        self.buffer_metadata.append(meta)

        # Persist
        self._save_buffer()

        print(f"RealTimeIndex: added product ID={new_id} '{name}' — buffer size: {len(self.buffer_embeddings)}")
        return meta

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        candidates: Optional[List[int]] = None
    ) -> List[Tuple[dict, float]]:
        """
        Searches both the main FAISS index and the live buffer.
        Returns merged top-k results as [(metadata_dict, score), ...].
        """
        results = []

        # Search main FAISS index
        if self.faiss_index and self.faiss_index.index:
            try:
                faiss_results = self.faiss_index.search(query_vec, candidates, k)
                results.extend(faiss_results)
            except:
                pass

        # Search live buffer (exact brute-force — buffer is small)
        if self.buffer_embeddings:
            buf_matrix = np.array(self.buffer_embeddings)  # (B, 512)
            scores     = buf_matrix @ query_vec             # (B,)
            top_buf    = np.argsort(scores)[-k:][::-1]

            for buf_idx in top_buf:
                buf_score = float(scores[buf_idx])
                buf_meta  = self.buffer_metadata[buf_idx].copy()
                # Use negative index to distinguish buffer from main index
                results.append((-(buf_idx + 1), buf_score, buf_meta))

        # Sort all results by score
        results_with_meta = []
        for item in results:
            if len(item) == 2:
                idx, score = item
                results_with_meta.append({"idx": idx, "score": score, "is_buffer": False})
            elif len(item) == 3 and isinstance(item[2], dict):
                idx, score, meta = item
                results_with_meta.append({"idx": idx, "score": score, "is_buffer": True, "meta": meta})
            else:
                idx, score = item[0], item[1]
                results_with_meta.append({"idx": idx, "score": score, "is_buffer": False})

        results_with_meta.sort(key=lambda x: x["score"], reverse=True)
        return results_with_meta[:k]

    def buffer_products(self) -> List[dict]:
        """Returns all buffered (real-time added) products."""
        return [
            {**meta, "embedding_norm": round(float(np.linalg.norm(emb)), 4)}
            for meta, emb in zip(self.buffer_metadata, self.buffer_embeddings)
        ]

    def clear_buffer(self):
        """Clears all real-time added products."""
        self.buffer_embeddings = []
        self.buffer_metadata   = []
        self._save_buffer()
        print("RealTimeIndex: buffer cleared")

    @property
    def total_products(self) -> int:
        main = self.faiss_index.index.ntotal if (self.faiss_index and self.faiss_index.index) else 0
        return main + len(self.buffer_embeddings)

    @property
    def buffer_size(self) -> int:
        return len(self.buffer_embeddings)
