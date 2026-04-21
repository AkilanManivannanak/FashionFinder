"""
engine/hash_index.py
---------------------
DATA STRUCTURE: Hash Table
--------------------------
Indexes all products by masterCategory (Apparel, Footwear, Accessories, etc.)
so that when a query comes in, we only search within the matching category bucket
instead of scanning the entire dataset.

This cuts the search space by roughly 60-80% on average.

Structure:
    {
        "Apparel":     [0, 3, 7, 12, ...],   # list of row indices into metadata
        "Footwear":    [1, 4, 9, ...],
        "Accessories": [2, 5, 6, ...],
        ...
    }
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class HashIndex:
    """
    Hash table mapping category -> list of product indices.
    Enables O(1) category lookup and dramatically reduces search space.
    """

    def __init__(self, metadata: pd.DataFrame):
        """
        Builds the hash table from the metadata DataFrame.
        Key   = masterCategory string
        Value = list of integer row indices
        """
        self._table: Dict[str, List[int]] = {}
        self._build(metadata)

    def _build(self, metadata: pd.DataFrame):
        """Iterates once over metadata to populate the hash table."""
        for idx, row in metadata.iterrows():
            category = str(row.get("masterCategory", "Unknown")).strip()
            if category not in self._table:
                self._table[category] = []
            self._table[category].append(idx)

        print(f"HashIndex built: {len(self._table)} categories")
        for cat, indices in sorted(self._table.items()):
            print(f"  {cat:30s}: {len(indices):>6,} products")

    def get_indices(self, category: str) -> List[int]:
        """
        Returns all row indices for a given category.
        Falls back to ALL indices if category not found.
        """
        if category in self._table:
            return self._table[category]
        # Fallback: search everything
        all_indices = []
        for v in self._table.values():
            all_indices.extend(v)
        return all_indices

    def categories(self) -> List[str]:
        """Returns all known category keys."""
        return list(self._table.keys())

    def size(self, category: str = None) -> int:
        """Returns number of products in a category, or total if None."""
        if category:
            return len(self._table.get(category, []))
        return sum(len(v) for v in self._table.values())

    def infer_category(self, metadata: pd.DataFrame, query_idx: int) -> str:
        """
        Given a product index, returns its masterCategory.
        Used to automatically pick the right bucket for a query image
        when searching by example from within the dataset.
        """
        try:
            return str(metadata.loc[query_idx, "masterCategory"]).strip()
        except Exception:
            return "Unknown"
