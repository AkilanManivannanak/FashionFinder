"""
brand_index.py
--------------
DATA STRUCTURE: Three-Level Nested Hash Table
----------------------------------------------
Extends the color index with a third level:
    brand -> category -> [product indices]
    category -> brand -> [product indices]

Also provides:
    brand -> all indices (flat lookup)
    top brands by product count

Brand extraction: parse productDisplayName to extract known brands.
Uses a curated brand list + fuzzy matching for unknown names.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter


# Known major fashion brands to extract from product names
KNOWN_BRANDS = [
    # Sportswear
    "Nike", "Adidas", "Puma", "Reebok", "New Balance", "Under Armour",
    "FILA", "Asics", "Skechers", "Converse", "Vans", "Columbia",
    "The North Face", "Quiksilver", "Billabong",
    # Indian / Kaggle dataset specific brands
    "Mumbai Indians", "Kochi Tuskers", "Tantra", "Gini", "Jony",
    "Fabindia", "FabIndia", "Prafful", "Doodle", "Aneri", "Aurelia",
    "Lee", "Wrangler", "Levi", "Levis", "Flying Machine", "Pepe Jeans",
    "United Colors", "Benetton", "Jack Jones", "Only", "Vero Moda",
    "Allen Solly", "Van Heusen", "Arrow", "Raymond", "Peter England",
    "Blackberrys", "ColorPlus", "Park Avenue",
    # Footwear
    "Woodland", "Bata", "Liberty", "Red Tape", "Kalenji", "Quechua",
    "Sparx", "Campus", "Action", "Paragon",
    # Accessories
    "Fastrack", "Titan", "Casio", "Timex", "Citizen",
    # General
    "H&M", "Zara", "Mango", "Forever 21", "Marks Spencer",
    "Tommy Hilfiger", "Calvin Klein", "Ralph Lauren", "Guess",
    "Fossil", "Lacoste", "Polo",
]

# Sort by length descending so longer brands match first (e.g. "New Balance" before "New")
KNOWN_BRANDS_SORTED = sorted(KNOWN_BRANDS, key=len, reverse=True)


def extract_brand(product_name: str) -> str:
    """
    Extracts brand from product display name.
    Strategy:
      1. Check against known brands list (case-insensitive)
      2. Fall back to first word(s) of product name
    """
    if not product_name or product_name == "nan":
        return "Unknown"

    name_lower = product_name.lower()

    for brand in KNOWN_BRANDS_SORTED:
        if brand.lower() in name_lower:
            return brand

    # Fallback: use first word as brand (e.g. "Reebok Women Black..." -> "Reebok")
    first_word = product_name.strip().split()[0] if product_name.strip() else "Unknown"
    # Filter out common non-brand first words
    skip_words = {"the", "a", "an", "men", "women", "boys", "girls", "kids", "unisex"}
    if first_word.lower() in skip_words and len(product_name.split()) > 1:
        first_word = product_name.strip().split()[1]

    return first_word


class BrandIndex:
    """
    Three-level nested hash table:
        brand_to_indices:    brand -> [row indices]
        brand_to_categories: brand -> {category -> [indices]}
        category_to_brands:  category -> {brand -> [indices]}

    Enables queries like:
        "Give me all Nike Footwear products"
        "What brands are available in Apparel?"
        "Find the top 10 brands by product count"
    """

    def __init__(self, metadata: pd.DataFrame):
        self.brand_to_indices: Dict[str, List[int]] = {}
        self.brand_to_categories: Dict[str, Dict[str, List[int]]] = {}
        self.category_to_brands: Dict[str, Dict[str, List[int]]] = {}
        self.product_brands: List[str] = []  # brand per row index
        self._build(metadata)

    def _build(self, metadata: pd.DataFrame):
        for idx, row in metadata.iterrows():
            name = str(row.get("productDisplayName", ""))
            cat  = str(row.get("masterCategory", "Unknown")).strip()
            brand = extract_brand(name)

            self.product_brands.append(brand)

            # Flat: brand -> indices
            if brand not in self.brand_to_indices:
                self.brand_to_indices[brand] = []
            self.brand_to_indices[brand].append(idx)

            # brand -> category -> indices
            if brand not in self.brand_to_categories:
                self.brand_to_categories[brand] = {}
            if cat not in self.brand_to_categories[brand]:
                self.brand_to_categories[brand][cat] = []
            self.brand_to_categories[brand][cat].append(idx)

            # category -> brand -> indices
            if cat not in self.category_to_brands:
                self.category_to_brands[cat] = {}
            if brand not in self.category_to_brands[cat]:
                self.category_to_brands[cat][brand] = []
            self.category_to_brands[cat][brand].append(idx)

        print(f"BrandIndex built: {len(self.brand_to_indices)} brands across {len(self.category_to_brands)} categories")

    def get_indices(
        self,
        brand: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[int]:
        """Returns indices matching brand and/or category filters."""
        if brand and category:
            return self.brand_to_categories.get(brand, {}).get(category, [])
        elif brand:
            return self.brand_to_indices.get(brand, [])
        elif category:
            all_idx = []
            for indices in self.category_to_brands.get(category, {}).values():
                all_idx.extend(indices)
            return all_idx
        else:
            all_idx = []
            for indices in self.brand_to_indices.values():
                all_idx.extend(indices)
            return all_idx

    def get_brand(self, idx: int) -> str:
        """Returns the brand for a given product row index."""
        if idx < len(self.product_brands):
            return self.product_brands[idx]
        return "Unknown"

    def top_brands(self, category: Optional[str] = None, n: int = 20) -> List[Tuple[str, int]]:
        """Returns top-n brands by product count, optionally within a category."""
        if category:
            brand_counts = {
                brand: len(indices)
                for brand, indices in self.category_to_brands.get(category, {}).items()
            }
        else:
            brand_counts = {
                brand: len(indices)
                for brand, indices in self.brand_to_indices.items()
            }
        return sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def brands_in_category(self, category: str) -> List[str]:
        """Returns all brands available in a category."""
        return sorted(self.category_to_brands.get(category, {}).keys())

    def categories_for_brand(self, brand: str) -> List[str]:
        """Returns all categories a brand appears in."""
        return sorted(self.brand_to_categories.get(brand, {}).keys())

    def all_brands(self) -> List[str]:
        """Returns all known brands sorted alphabetically."""
        return sorted(self.brand_to_indices.keys())

    def summary(self) -> List[dict]:
        """Full brand summary for API/UI."""
        rows = []
        for brand, indices in sorted(self.brand_to_indices.items(), key=lambda x: -len(x[1])):
            cats = list(self.brand_to_categories.get(brand, {}).keys())
            rows.append({
                "brand": brand,
                "count": len(indices),
                "categories": cats
            })
        return rows
