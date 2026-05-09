"""
explainer.py
------------
Explains WHY a product matched a query.

Given a query product and a result product, generates a human-readable
explanation like:
    "Matched because: Same category (Tshirts), Same color (Blue),
     Same brand family (Nike), High visual similarity (Score: 0.912)"

Uses metadata comparison + embedding similarity breakdown.
No additional ML needed — pure metadata + score analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class SimilarityExplainer:
    """
    Generates human-readable explanations for why a result matched a query.
    Works by comparing metadata fields between query and result.
    """

    def __init__(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        self.metadata   = metadata
        self.embeddings = embeddings

    def explain(
        self,
        query_idx: Optional[int],
        result_idx: int,
        score: float,
        query_meta: Optional[dict] = None
    ) -> dict:
        """
        Generates explanation for a single result.

        Args:
            query_idx:  row index of query product (None for uploaded images)
            result_idx: row index of result product
            score:      cosine similarity score
            query_meta: optional dict with query metadata fields

        Returns:
            dict with explanation fields
        """
        result_row = self.metadata.loc[result_idx]

        # Get query metadata
        if query_idx is not None and query_idx < len(self.metadata):
            query_row = self.metadata.loc[query_idx]
            q_cat    = str(query_row.get("masterCategory", "")).strip()
            q_sub    = str(query_row.get("subCategory", "")).strip()
            q_art    = str(query_row.get("articleType", "")).strip()
            q_color  = str(query_row.get("baseColour", "")).strip()
            q_season = str(query_row.get("season", "")).strip()
        elif query_meta:
            q_cat    = query_meta.get("masterCategory", "")
            q_sub    = query_meta.get("subCategory", "")
            q_art    = query_meta.get("articleType", "")
            q_color  = query_meta.get("baseColour", "")
            q_season = query_meta.get("season", "")
        else:
            q_cat = q_sub = q_art = q_color = q_season = ""

        r_cat    = str(result_row.get("masterCategory", "")).strip()
        r_sub    = str(result_row.get("subCategory", "")).strip()
        r_art    = str(result_row.get("articleType", "")).strip()
        r_color  = str(result_row.get("baseColour", "")).strip()
        r_season = str(result_row.get("season", "")).strip()

        reasons = []
        match_score = 0.0

        # Article type match (strongest signal)
        if q_art and r_art and q_art.lower() == r_art.lower():
            reasons.append(f"Same type ({r_art})")
            match_score += 0.35

        # Category match
        if q_cat and r_cat and q_cat.lower() == r_cat.lower():
            reasons.append(f"Same category ({r_cat})")
            match_score += 0.20

        # Sub-category match
        if q_sub and r_sub and q_sub.lower() == r_sub.lower():
            reasons.append(f"Same sub-category ({r_sub})")
            match_score += 0.15

        # Color match
        if q_color and r_color and q_color.lower() == r_color.lower():
            reasons.append(f"Same color ({r_color})")
            match_score += 0.20
        elif q_color and r_color:
            # Check for color family similarity
            color_families = {
                "dark": ["black", "navy blue", "dark blue", "charcoal", "dark brown"],
                "light": ["white", "cream", "off white", "beige", "ivory"],
                "blue_family": ["blue", "navy blue", "teal", "cobalt"],
                "red_family": ["red", "maroon", "burgundy", "rust"],
            }
            for family, colors in color_families.items():
                if q_color.lower() in colors and r_color.lower() in colors:
                    reasons.append(f"Similar color family ({q_color} ~ {r_color})")
                    match_score += 0.10
                    break

        # Season match
        if q_season and r_season and q_season.lower() == r_season.lower():
            reasons.append(f"Same season ({r_season})")
            match_score += 0.10

        # Visual similarity score interpretation
        if score >= 0.95:
            reasons.append(f"Visually near-identical (score: {score:.3f})")
        elif score >= 0.90:
            reasons.append(f"Very high visual similarity (score: {score:.3f})")
        elif score >= 0.85:
            reasons.append(f"High visual similarity (score: {score:.3f})")
        elif score >= 0.75:
            reasons.append(f"Good visual similarity (score: {score:.3f})")
        else:
            reasons.append(f"Moderate visual similarity (score: {score:.3f})")

        if not reasons:
            reasons = [f"Visual embedding similarity (score: {score:.3f})"]

        return {
            "score": round(score, 4),
            "reasons": reasons,
            "match_summary": " · ".join(reasons[:3]),
            "article_match":  q_art == r_art and bool(q_art),
            "color_match":    q_color == r_color and bool(q_color),
            "category_match": q_cat == r_cat and bool(q_cat),
        }

    def explain_batch(
        self,
        query_idx: Optional[int],
        results: List[Tuple[int, float]],
        query_meta: Optional[dict] = None
    ) -> List[dict]:
        """Explains a list of (result_idx, score) pairs."""
        return [
            self.explain(query_idx, idx, score, query_meta)
            for idx, score in results
        ]
