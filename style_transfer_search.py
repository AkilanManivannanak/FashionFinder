"""
style_transfer_search.py
Fixed: fuzzy color matching so "Copper", "copper", "COPPER" all work
"""

import numpy as np
from typing import List, Tuple, Optional
from heap_ranker import top_k_cosine


class StyleTransferSearch:
    def __init__(self, embeddings, metadata, color_index, brand_index=None):
        self.embeddings  = embeddings
        self.metadata    = metadata
        self.color_index = color_index
        self.brand_index = brand_index
        # Build lowercase color lookup once
        self._all_colors = color_index.colors(None)
        self._color_lower = {c.lower(): c for c in self._all_colors}

    def _resolve_color(self, target_color: str, category: str = None) -> Optional[str]:
        """
        Resolves a color string to the exact color name in the index.
        Handles:
          - Exact match: "Blue" -> "Blue"
          - Case insensitive: "blue" -> "Blue"
          - Partial match: "cop" -> "Copper"
          - Close match: "Navy" -> "Navy Blue"
        Returns the resolved color or None if not found.
        """
        if not target_color:
            return None

        # Get colors available for this category
        if category:
            avail = self.color_index.colors(category)
        else:
            avail = self._all_colors

        if not avail:
            return None

        avail_lower = {c.lower(): c for c in avail}

        # 1. Exact match
        if target_color in avail:
            return target_color

        # 2. Case-insensitive exact match
        tl = target_color.lower()
        if tl in avail_lower:
            return avail_lower[tl]

        # 3. Starts-with match (e.g. "cop" -> "Copper")
        matches = [c for cl, c in avail_lower.items() if cl.startswith(tl)]
        if matches:
            return matches[0]

        # 4. Contains match (e.g. "oppe" -> "Copper")
        matches = [c for cl, c in avail_lower.items() if tl in cl]
        if matches:
            return matches[0]

        # 5. Target contains color (e.g. "Navy Blue" -> "Navy Blue" or "Blue")
        matches = [c for cl, c in avail_lower.items() if cl in tl]
        if matches:
            # Pick longest match
            return max(matches, key=len)

        # 6. Word overlap (e.g. "Copper Brown" -> "Copper" or "Brown")
        target_words = set(tl.split())
        best = None
        best_overlap = 0
        for cl, c in avail_lower.items():
            color_words = set(cl.split())
            overlap = len(target_words & color_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best = c
        if best:
            return best

        return None

    def search(
        self,
        query_vec: np.ndarray,
        target_color: str,
        target_category: Optional[str] = None,
        k: int = 10,
        query_idx: Optional[int] = None
    ) -> List[dict]:
        # Resolve color with fuzzy matching
        resolved_color = self._resolve_color(target_color, target_category)

        if not resolved_color:
            # Fall back to full category search if color not found
            candidates = self.color_index.get_indices(category=target_category) if target_category else list(range(len(self.embeddings)))
            resolved_color = target_color  # Keep original for display
        else:
            candidates = self.color_index.get_indices(
                category=target_category,
                color=resolved_color
            )

        if not candidates:
            return []

        # Exclude query itself
        if query_idx is not None:
            candidates = [i for i in candidates if i != query_idx]

        if not candidates:
            return []

        top_k = top_k_cosine(query_vec, self.embeddings, candidates, k)

        results = []
        for i, (idx, score) in enumerate(top_k):
            row = self.metadata.loc[idx]
            pid = int(row.get("id", idx))
            brand = self.brand_index.get_brand(idx) if self.brand_index else "Unknown"
            results.append({
                "rank":           i + 1,
                "product_idx":    int(idx),
                "id":             pid,
                "name":           str(row.get("productDisplayName", "Unknown")),
                "masterCategory": str(row.get("masterCategory", "Unknown")),
                "articleType":    str(row.get("articleType", "Unknown")),
                "baseColour":     str(row.get("baseColour", "Unknown")),
                "season":         str(row.get("season", "Unknown")),
                "brand":          brand,
                "score":          round(float(score), 4),
                "image_path":     f"archive-2/images/{pid}.jpg",
                "explanation":    f"Same style in {resolved_color} (score: {score:.3f})",
                "resolved_color": resolved_color,
            })

        return results

    def color_variants(
        self,
        query_vec: np.ndarray,
        category: Optional[str] = None,
        k_per_color: int = 1,
        query_idx: Optional[int] = None,
        max_colors: int = 10
    ) -> dict:
        """
        Finds the best match for this style across ALL colors simultaneously.
        Returns a color-keyed dict of results.
        """
        colors = self.color_index.colors(category)[:max_colors]
        results = {}

        for color in colors:
            color_results = self.search(
                query_vec, color, category, k_per_color, query_idx
            )
            if color_results:
                results[color] = color_results

        return results
