"""
trend_tracker.py
----------------
Tracks search history and computes trending products.
Stores every search query in a SQLite database with timestamp.
Computes trending products by search frequency over a time window.

No external dependencies beyond stdlib + pandas.
"""

import sqlite3
import os
import time
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd


class TrendTracker:
    """
    Logs every search and computes trending products over time windows.
    Uses SQLite for zero-infra persistence.
    """

    def __init__(self, db_path: str = "data/trends.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL NOT NULL,
                    query_type  TEXT,
                    method      TEXT,
                    category    TEXT,
                    color       TEXT,
                    brand       TEXT,
                    result_ids  TEXT,
                    latency_ms  REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS product_views (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL NOT NULL,
                    product_idx INTEGER NOT NULL,
                    product_id  INTEGER,
                    rank        INTEGER,
                    score       REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON product_views(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pid ON product_views(product_idx)")
            conn.commit()

    def log_search(
        self,
        query_type: str,
        method: str,
        results: List[dict],
        latency_ms: float,
        category: str = None,
        color: str = None,
        brand: str = None
    ):
        """Logs a search and all result product views."""
        ts = time.time()
        result_ids = ",".join(str(r.get("product_idx", "")) for r in results)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO search_log (timestamp, query_type, method, category, color, brand, result_ids, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, query_type, method, category, color, brand, result_ids, latency_ms))

            for r in results:
                conn.execute("""
                    INSERT INTO product_views (timestamp, product_idx, product_id, rank, score)
                    VALUES (?, ?, ?, ?, ?)
                """, (ts, r.get("product_idx", -1), r.get("id", -1), r.get("rank", 0), r.get("score", 0)))
            conn.commit()

    def trending_products(
        self,
        hours: int = 24,
        top_n: int = 20
    ) -> List[Tuple[int, int]]:
        """
        Returns top-n most-viewed product indices in the last `hours` hours.
        Returns [(product_idx, view_count), ...] sorted descending.
        """
        cutoff = time.time() - hours * 3600
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT product_idx, COUNT(*) as views
                FROM product_views
                WHERE timestamp > ?
                GROUP BY product_idx
                ORDER BY views DESC
                LIMIT ?
            """, (cutoff, top_n)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def trending_categories(self, hours: int = 24) -> List[Tuple[str, int]]:
        """Returns most-searched categories in the last N hours."""
        cutoff = time.time() - hours * 3600
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT category, COUNT(*) as searches
                FROM search_log
                WHERE timestamp > ? AND category IS NOT NULL
                GROUP BY category
                ORDER BY searches DESC
            """, (cutoff,)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def trending_brands(self, hours: int = 24) -> List[Tuple[str, int]]:
        """Returns most-searched brands in the last N hours."""
        cutoff = time.time() - hours * 3600
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT brand, COUNT(*) as searches
                FROM search_log
                WHERE timestamp > ? AND brand IS NOT NULL
                GROUP BY brand
                ORDER BY searches DESC
            """, (cutoff,)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def total_searches(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]

    def search_history(self, limit: int = 50) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT timestamp, query_type, method, category, color, brand, latency_ms
                FROM search_log ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        return [
            {
                "time": datetime.fromtimestamp(r[0]).strftime("%H:%M:%S"),
                "query_type": r[1], "method": r[2],
                "category": r[3], "color": r[4],
                "brand": r[5], "latency_ms": r[6]
            }
            for r in rows
        ]
