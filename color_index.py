"""
color_index.py
--------------
DATA STRUCTURE: Nested Hash Table
-----------------------------------
Extends the category hash index with a second level:
    category -> colour -> [product indices]

Enables combined filtering:
    "Show me blue Apparel only" -> hash_table["Apparel"]["Blue"]

This is a nested hash table / two-level index, reducing
search space further beyond category alone.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple


class ColorIndex:
    """
    Two-level hash table: category -> colour -> list of indices.
    Also provides a flat colour -> indices lookup.
    """

    def __init__(self, metadata: pd.DataFrame):
        # Two-level: category -> colour -> indices
        self._cat_color: Dict[str, Dict[str, List[int]]] = {}
        # Flat: colour -> indices
        self._color_flat: Dict[str, List[int]] = {}
        self._build(metadata)

    def _build(self, metadata: pd.DataFrame):
        for idx, row in metadata.iterrows():
            cat   = str(row.get("masterCategory", "Unknown")).strip()
            color = str(row.get("baseColour", "Unknown")).strip()

            # Two-level
            if cat not in self._cat_color:
                self._cat_color[cat] = {}
            if color not in self._cat_color[cat]:
                self._cat_color[cat][color] = []
            self._cat_color[cat][color].append(idx)

            # Flat
            if color not in self._color_flat:
                self._color_flat[color] = []
            self._color_flat[color].append(idx)

        total_colors = len(self._color_flat)
        print(f"ColorIndex built: {total_colors} unique colours across all categories")

    def get_indices(
        self,
        category: Optional[str] = None,
        color: Optional[str] = None
    ) -> List[int]:
        """
        Returns indices matching both category and color filters.
        Falls back gracefully if either filter is missing.
        """
        if category and color:
            return self._cat_color.get(category, {}).get(color, [])
        elif color:
            return self._color_flat.get(color, [])
        elif category:
            # Flatten all colours for this category
            all_idx = []
            for indices in self._cat_color.get(category, {}).values():
                all_idx.extend(indices)
            return all_idx
        else:
            # No filter: return everything
            all_idx = []
            for indices in self._color_flat.values():
                all_idx.extend(indices)
            return all_idx

    def colors(self, category: Optional[str] = None) -> List[str]:
        """Returns all known colours, optionally filtered by category."""
        if category:
            return sorted(self._cat_color.get(category, {}).keys())
        return sorted(self._color_flat.keys())

    def categories(self) -> List[str]:
        return sorted(self._cat_color.keys())

    def summary(self) -> List[dict]:
        """Returns a summary list for API/UI display."""
        rows = []
        for cat, colors in sorted(self._cat_color.items()):
            for color, indices in sorted(colors.items()):
                rows.append({
                    "category": cat,
                    "color": color,
                    "count": len(indices)
                })
        return rows
