"""
fashion_timeline.py
-------------------
Fashion Timeline: How fashion trends evolved year by year.

Analyzes the dataset across years to show:
- Which article types peaked in which year
- Color trend evolution (what colors were popular each year)
- Brand growth over time
- Category distribution changes
- Visual similarity clusters per year
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


class FashionTimeline:
    """
    Analyzes fashion trends across years in the dataset.
    Provides year-by-year breakdown of styles, colors, and brands.
    """

    def __init__(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        self.metadata   = metadata
        self.embeddings = embeddings
        self._prepare()

    def _prepare(self):
        """Pre-processes year data."""
        self.df = self.metadata.copy()
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce")
        self.df = self.df.dropna(subset=["year"])
        self.df["year"] = self.df["year"].astype(int)
        self.years = sorted(self.df["year"].unique())
        print(f"FashionTimeline: {len(self.years)} years from {min(self.years)} to {max(self.years)}")

    def color_trends_by_year(self, top_n: int = 8) -> Dict[int, List[Tuple[str, int]]]:
        """Returns top-n colors per year."""
        result = {}
        for year in self.years:
            year_df = self.df[self.df["year"] == year]
            color_counts = Counter(year_df["baseColour"].dropna())
            result[year] = color_counts.most_common(top_n)
        return result

    def article_trends_by_year(self, top_n: int = 8) -> Dict[int, List[Tuple[str, int]]]:
        """Returns top-n article types per year."""
        result = {}
        for year in self.years:
            year_df = self.df[self.df["year"] == year]
            art_counts = Counter(year_df["articleType"].dropna())
            result[year] = art_counts.most_common(top_n)
        return result

    def brand_trends_by_year(self, brand_index, top_n: int = 5) -> Dict[int, List[Tuple[str, int]]]:
        """Returns top-n brands per year using brand_index."""
        result = {}
        for year in self.years:
            year_indices = self.df[self.df["year"] == year].index.tolist()
            brand_counts = Counter(
                brand_index.get_brand(idx) for idx in year_indices
            )
            result[year] = brand_counts.most_common(top_n)
        return result

    def category_distribution_by_year(self) -> Dict[int, Dict[str, int]]:
        """Returns category distribution per year."""
        result = {}
        for year in self.years:
            year_df = self.df[self.df["year"] == year]
            cat_counts = Counter(year_df["masterCategory"].dropna())
            result[year] = dict(cat_counts)
        return result

    def volume_by_year(self) -> List[Tuple[int, int]]:
        """Returns (year, product_count) pairs."""
        return [(year, int(self.df[self.df["year"] == year].shape[0])) for year in self.years]

    def plot_color_trends(self, top_colors: int = 6) -> bytes:
        """Plots color trend evolution as a stacked area chart."""
        # Get top colors overall
        all_color_counts = Counter(self.df["baseColour"].dropna())
        top_color_names  = [c for c, _ in all_color_counts.most_common(top_colors)]

        year_data = {color: [] for color in top_color_names}
        for year in self.years:
            year_df = self.df[self.df["year"] == year]
            total   = max(len(year_df), 1)
            color_counts = Counter(year_df["baseColour"].dropna())
            for color in top_color_names:
                year_data[color].append(color_counts.get(color, 0) / total * 100)

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0e1117")
        ax.set_facecolor("#1e1e2e")

        palette = ["#4FC3F7", "#81C784", "#F48FB1", "#FFD700",
                   "#CE93D8", "#FFB74D", "#80DEEA", "#FF8A65"]

        for i, (color_name, values) in enumerate(year_data.items()):
            ax.plot(self.years, values, marker="o", linewidth=2,
                    color=palette[i % len(palette)], label=color_name)
            ax.fill_between(self.years, values, alpha=0.15, color=palette[i % len(palette)])

        ax.set_xlabel("Year", color="white")
        ax.set_ylabel("% of Products", color="white")
        ax.set_title("Fashion Color Trends Over Time", color="#4FC3F7", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e1e2e", labelcolor="white", loc="upper left")
        ax.spines[:].set_color("#444")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117", dpi=120)
        plt.close()
        buf.seek(0)
        return buf.read()

    def plot_article_trends(self, top_articles: int = 6) -> bytes:
        """Plots article type trend evolution."""
        all_art_counts  = Counter(self.df["articleType"].dropna())
        top_art_names   = [a for a, _ in all_art_counts.most_common(top_articles)]

        year_data = {art: [] for art in top_art_names}
        for year in self.years:
            year_df = self.df[self.df["year"] == year]
            total   = max(len(year_df), 1)
            art_counts = Counter(year_df["articleType"].dropna())
            for art in top_art_names:
                year_data[art].append(art_counts.get(art, 0) / total * 100)

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0e1117")
        ax.set_facecolor("#1e1e2e")

        palette = ["#4FC3F7", "#81C784", "#F48FB1", "#FFD700", "#CE93D8", "#FFB74D"]
        for i, (art_name, values) in enumerate(year_data.items()):
            ax.bar([y + i * 0.12 for y in range(len(self.years))], values,
                   width=0.12, color=palette[i % len(palette)],
                   label=art_name, alpha=0.85)

        ax.set_xticks(range(len(self.years)))
        ax.set_xticklabels(self.years, rotation=45, color="white")
        ax.set_ylabel("% of Products", color="white")
        ax.set_title("Fashion Article Type Trends Over Time", color="#4FC3F7", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1e1e2e", labelcolor="white", loc="upper right")
        ax.spines[:].set_color("#444")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117", dpi=120)
        plt.close()
        buf.seek(0)
        return buf.read()

    def plot_volume_by_year(self) -> bytes:
        """Plots product volume per year."""
        vols = self.volume_by_year()
        years  = [v[0] for v in vols]
        counts = [v[1] for v in vols]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
        ax.set_facecolor("#1e1e2e")

        bars = ax.bar(years, counts, color="#4FC3F7", alpha=0.85, width=0.6)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{count:,}", ha="center", va="bottom", color="white", fontsize=9)

        ax.set_xlabel("Year", color="white")
        ax.set_ylabel("Number of Products", color="white")
        ax.set_title("Fashion Dataset Volume by Year", color="#4FC3F7", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white", axis="x", rotation=45)
        ax.tick_params(colors="white", axis="y")
        ax.spines[:].set_color("#444")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117", dpi=120)
        plt.close()
        buf.seek(0)
        return buf.read()

    def year_summary(self, year: int) -> dict:
        """Returns a full summary for a specific year."""
        year_df = self.df[self.df["year"] == year]
        if year_df.empty:
            return {"error": f"No data for year {year}"}

        top_colors   = Counter(year_df["baseColour"].dropna()).most_common(5)
        top_articles = Counter(year_df["articleType"].dropna()).most_common(5)
        top_cats     = Counter(year_df["masterCategory"].dropna()).most_common(5)

        return {
            "year":         year,
            "total_products": len(year_df),
            "top_colors":   [{"color": c, "count": n} for c, n in top_colors],
            "top_articles": [{"type": t, "count": n} for t, n in top_articles],
            "top_categories": [{"category": c, "count": n} for c, n in top_cats],
        }
