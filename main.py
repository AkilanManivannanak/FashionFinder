"""
main.py  -  FashionFinder Advanced API
----------------------------------------
New endpoints added:
    GET  /colors                  - list colours (from color index)
    POST /search/upload           - now supports color filter + faiss method
    POST /search/by_id            - now supports color filter + faiss method
    GET  /graph_neighbors/{id}    - returns k-NN graph neighbors for visualization
    POST /benchmark               - runs all 3 methods over N random queries
"""

import os
import sys
import io
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder    import Embedder
from hash_index  import HashIndex
from color_index import ColorIndex
from knn_graph   import KNNGraph
from faiss_index import FAISSIndex
from searcher    import Searcher

# ── Paths ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = "embeddings/embeddings.npy"
METADATA_PATH   = "embeddings/metadata.csv"
GRAPH_PATH      = "embeddings/knn_graph.pkl"
FAISS_PATH      = "embeddings/faiss.index"
IMAGES_DIR      = "archive-2/images"

app = FastAPI(
    title="FashionFinder Advanced API",
    description="Visual search with Baseline, k-NN Graph, and FAISS ANN retrieval.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Load components ───────────────────────────────────────────────────────────
print("Loading FashionFinder components...")

embeddings  = np.load(EMBEDDINGS_PATH)
metadata    = pd.read_csv(METADATA_PATH).reset_index(drop=True)

embedder    = Embedder()
hash_index  = HashIndex(metadata)
color_index = ColorIndex(metadata)

knn_graph = KNNGraph()
if os.path.exists(GRAPH_PATH):
    knn_graph.load(GRAPH_PATH)
else:
    print("WARNING: k-NN graph not found.")
    knn_graph = None

faiss_index = FAISSIndex()
if os.path.exists(FAISS_PATH):
    faiss_index.load(FAISS_PATH)
else:
    print("WARNING: FAISS index not found. Run: python build_faiss.py")
    faiss_index = None

searcher = Searcher(embeddings, metadata, hash_index, knn_graph, IMAGES_DIR)
print(f"All components loaded. {len(embeddings):,} products indexed.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def idx_from_product_id(product_id: int) -> int:
    matches = metadata[metadata["id"] == product_id].index.tolist()
    if not matches:
        raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found.")
    return matches[0]

def get_candidates(category: Optional[str], color: Optional[str]) -> Optional[list]:
    """Returns candidate indices from color_index (supports both filters)."""
    if category or color:
        return color_index.get_indices(category=category, color=color)
    return None

def row_to_meta(idx: int) -> dict:
    row = metadata.loc[idx]
    pid = int(row.get("id", idx))
    return {
        "id":              pid,
        "name":            str(row.get("productDisplayName", "Unknown")),
        "masterCategory":  str(row.get("masterCategory", "Unknown")),
        "subCategory":     str(row.get("subCategory", "Unknown")),
        "articleType":     str(row.get("articleType", "Unknown")),
        "baseColour":      str(row.get("baseColour", "Unknown")),
        "season":          str(row.get("season", "Unknown")),
        "year":            str(row.get("year", "Unknown")),
        "image_path":      f"{IMAGES_DIR}/{pid}.jpg"
    }

def run_faiss_search(query_vec, candidates, k):
    """Runs FAISS search and returns results list."""
    if faiss_index is None:
        return [], 0.0
    t0 = time.perf_counter()
    top_k = faiss_index.search(query_vec, candidates, k)
    latency = (time.perf_counter() - t0) * 1000
    results = []
    for i, (idx, score) in enumerate(top_k):
        try:
            if idx < 0 or idx >= len(embeddings):
                continue
            meta = row_to_meta(idx)
            meta["rank"] = i + 1
            meta["product_idx"] = idx
            meta["score"] = round(score, 4)
            results.append(meta)
        except Exception as e:
            print(f"Skipping idx {idx}: {e}")
            continue
    return results, round(latency, 2)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":           "ok",
        "products_indexed": len(embeddings),
        "categories":       hash_index.categories(),
        "graph_loaded":     knn_graph is not None,
        "faiss_loaded":     faiss_index is not None,
    }


@app.get("/categories")
def get_categories():
    return {
        "categories": [
            {"name": cat, "count": hash_index.size(cat)}
            for cat in sorted(hash_index.categories())
        ]
    }


@app.get("/colors")
def get_colors(category: Optional[str] = None):
    return {"colors": color_index.colors(category)}


@app.post("/search/upload")
async def search_by_upload(
    file: UploadFile = File(...),
    k:        int            = Query(default=10, ge=1, le=50),
    method:   str            = Query(default="graph"),
    category: Optional[str]  = Query(default=None),
    color:    Optional[str]  = Query(default=None)
):
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    query_vec  = embedder.embed_pil(pil_img)
    candidates = get_candidates(category, color)

    if method == "faiss":
        results, latency = run_faiss_search(query_vec, candidates, k)
        return {"method": "faiss", "latency_ms": latency, "k": k,
                "category_filter": category, "color_filter": color, "results": results}

    # Use existing searcher for baseline/graph
    cand_list = candidates if candidates else list(range(len(embeddings)))
    return searcher.search(query_vec, k=k, method=method,
                           category=category, query_idx=None)


@app.post("/search/by_id")
def search_by_id(
    product_id: int,
    k:        int            = Query(default=10, ge=1, le=50),
    method:   str            = Query(default="graph"),
    category: Optional[str]  = Query(default=None),
    color:    Optional[str]  = Query(default=None)
):
    query_idx  = idx_from_product_id(product_id)
    query_vec  = embeddings[query_idx]
    candidates = get_candidates(category, color)

    if not category:
        category = hash_index.infer_category(metadata, query_idx)

    if method == "faiss":
        cands = candidates if candidates else None
        results, latency = run_faiss_search(query_vec, None, k + 1)
        # Exclude self
        results = [r for r in results if r.get("product_idx") != query_idx][:k]
        return {"method": "faiss", "latency_ms": latency, "k": k,
                "category_filter": category, "color_filter": color, "results": results}

    return searcher.search(query_vec, k=k, method=method,
                           category=category, query_idx=query_idx)


@app.get("/product/{product_id}")
def get_product(product_id: int):
    idx = idx_from_product_id(product_id)
    return row_to_meta(idx)


@app.get("/image/{product_id}")
def get_image(product_id: int):
    img_path = os.path.join(IMAGES_DIR, f"{product_id}.jpg")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/graph_neighbors/{product_id}")
def graph_neighbors(product_id: int, hops: int = Query(default=2, ge=1, le=3)):
    """
    Returns the k-NN graph neighborhood around a product for visualization.
    Performs BFS up to `hops` levels deep.
    """
    if knn_graph is None:
        raise HTTPException(status_code=503, detail="k-NN graph not loaded.")

    center_idx = idx_from_product_id(product_id)
    center_meta = row_to_meta(center_idx)

    nodes  = []
    visited = {center_idx}
    frontier = [(center_idx, 0)]  # (idx, depth)

    while frontier:
        curr_idx, depth = frontier.pop(0)
        if depth >= hops:
            continue

        neighbors = knn_graph.neighbors(curr_idx)
        for n_idx, n_score in neighbors[:8]:  # limit to 8 per node for readability
            if n_idx not in visited:
                visited.add(n_idx)
                meta = row_to_meta(n_idx)
                nodes.append({
                    **meta,
                    "score":  round(n_score, 4),
                    "depth":  depth + 1,
                    "parent": curr_idx
                })
                frontier.append((n_idx, depth + 1))

    return {
        "center": {**center_meta, "depth": 0, "parent": None},
        "nodes":  nodes,
        "total_nodes": len(nodes) + 1,
        "hops": hops
    }


@app.post("/benchmark")
def run_benchmark(n: int = Query(default=50, ge=5, le=500), k: int = Query(default=10)):
    """
    Runs all 3 retrieval methods over n random queries.
    Returns latency and recall statistics for the benchmark charts.
    """
    n = min(n, len(embeddings))
    query_indices = random.sample(range(len(embeddings)), n)

    b_lats, g_lats, f_lats = [], [], []
    g_recalls, f_recalls   = [], []

    for q_idx in query_indices:
        query_vec = embeddings[q_idx]
        category  = hash_index.infer_category(metadata, q_idx)

        # Baseline
        b_results, b_lat = searcher.search_baseline(
            query_vec, k=k, category=category, query_idx=q_idx)
        b_lats.append(b_lat)
        b_ids = set(r.product_idx for r in b_results)

        # Graph
        if knn_graph:
            g_results, g_lat = searcher.search_graph(
                query_vec, k=k, category=category, query_idx=q_idx)
            g_lats.append(g_lat)
            g_ids = set(r.product_idx for r in g_results)
            g_recalls.append(len(b_ids & g_ids) / len(b_ids) if b_ids else 0)

        # FAISS
        if faiss_index:
            t0 = time.perf_counter()
            f_top = faiss_index.search(
                query_vec,
                color_index.get_indices(category=category),
                k
            )
            f_lat = (time.perf_counter() - t0) * 1000
            f_lats.append(f_lat)
            f_ids = set(idx for idx, _ in f_top)
            f_recalls.append(len(b_ids & f_ids) / len(b_ids) if b_ids else 0)

    def p(arr, pct):
        return round(float(np.percentile(arr, pct)), 2) if arr else 0.0

    return {
        "n_queries":          n,
        "k":                  k,
        "baseline_median_ms": p(b_lats, 50),
        "baseline_p95_ms":    p(b_lats, 95),
        "baseline_p99_ms":    p(b_lats, 99),
        "graph_median_ms":    p(g_lats, 50),
        "graph_p95_ms":       p(g_lats, 95),
        "graph_p99_ms":       p(g_lats, 99),
        "faiss_median_ms":    p(f_lats, 50),
        "faiss_p95_ms":       p(f_lats, 95),
        "faiss_p99_ms":       p(f_lats, 99),
        "graph_recall":       round(float(np.mean(g_recalls)), 4) if g_recalls else 0,
        "faiss_recall":       round(float(np.mean(f_recalls)), 4) if f_recalls else 0,
    }
