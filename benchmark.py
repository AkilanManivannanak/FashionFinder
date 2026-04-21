"""
eval/benchmark.py
------------------
Benchmarks Baseline vs Graph retrieval on:

1. Latency      - median and p95 over N queries (milliseconds)
2. Recall@k     - what fraction of baseline top-k appears in graph top-k
                  (baseline is treated as ground truth for fair comparison
                   since we have no explicit relevance labels)

Usage:
    cd fashionfinder
    python eval/benchmark.py --n_queries 200 --k 10
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embedder import Embedder
from hash_index import HashIndex
from knn_graph import KNNGraph
from searcher import Searcher


def recall_at_k(baseline_ids, graph_ids):
    """
    Recall@k: fraction of baseline top-k results also found in graph top-k.
    Treats baseline as approximate ground truth.
    """
    baseline_set = set(baseline_ids)
    graph_set    = set(graph_ids)
    overlap = len(baseline_set & graph_set)
    return overlap / len(baseline_set) if baseline_set else 0.0


def run_benchmark(n_queries: int = 200, k: int = 10):
    # ── Load components ───────────────────────────────────────────────────────
    print("Loading components...")
    embeddings = np.load("embeddings/embeddings.npy")
    metadata   = pd.read_csv("embeddings/metadata.csv").reset_index(drop=True)

    hash_index = HashIndex(metadata)

    knn_graph = KNNGraph()
    knn_graph.load("embeddings/knn_graph.pkl")

    searcher = Searcher(embeddings, metadata, hash_index, knn_graph)

    # ── Sample random query products ─────────────────────────────────────────
    n_queries = min(n_queries, len(embeddings))
    query_indices = random.sample(range(len(embeddings)), n_queries)

    baseline_latencies = []
    graph_latencies    = []
    recalls            = []

    print(f"\nRunning {n_queries} queries, k={k}...\n")

    for q_idx in query_indices:
        query_vec = embeddings[q_idx]
        category  = hash_index.infer_category(metadata, q_idx)

        # Baseline
        b_results, b_latency = searcher.search_baseline(
            query_vec, k=k, category=category, query_idx=q_idx)
        baseline_latencies.append(b_latency)

        # Graph
        g_results, g_latency = searcher.search_graph(
            query_vec, k=k, category=category, query_idx=q_idx)
        graph_latencies.append(g_latency)

        # Recall@k
        b_ids = [r.product_idx for r in b_results]
        g_ids = [r.product_idx for r in g_results]
        recalls.append(recall_at_k(b_ids, g_ids))

    # ── Report ────────────────────────────────────────────────────────────────
    baseline_latencies = np.array(baseline_latencies)
    graph_latencies    = np.array(graph_latencies)
    recalls            = np.array(recalls)

    speedup = np.median(baseline_latencies) / np.median(graph_latencies)

    print("=" * 55)
    print(f"  FASHIONFINDER BENCHMARK  |  n={n_queries}  k={k}")
    print("=" * 55)
    print(f"\n  {'Metric':<30} {'Baseline':>10} {'Graph':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Latency Median (ms)':<30} {np.median(baseline_latencies):>10.2f} {np.median(graph_latencies):>10.2f}")
    print(f"  {'Latency p95 (ms)':<30} {np.percentile(baseline_latencies,95):>10.2f} {np.percentile(graph_latencies,95):>10.2f}")
    print(f"  {'Latency p99 (ms)':<30} {np.percentile(baseline_latencies,99):>10.2f} {np.percentile(graph_latencies,99):>10.2f}")
    print(f"  {'Recall@k':<30} {'1.000':>10} {np.mean(recalls):>10.4f}")
    print(f"\n  Speedup (graph vs baseline): {speedup:.2f}x faster")
    print("=" * 55)

    # Save results CSV
    os.makedirs("eval", exist_ok=True)
    results_df = pd.DataFrame({
        "query_idx": query_indices,
        "baseline_latency_ms": baseline_latencies,
        "graph_latency_ms": graph_latencies,
        "recall_at_k": recalls
    })
    results_df.to_csv("eval/benchmark_results.csv", index=False)
    print(f"\nDetailed results saved to eval/benchmark_results.csv")

    return {
        "baseline_median_ms": round(float(np.median(baseline_latencies)), 2),
        "graph_median_ms":    round(float(np.median(graph_latencies)), 2),
        "baseline_p95_ms":    round(float(np.percentile(baseline_latencies, 95)), 2),
        "graph_p95_ms":       round(float(np.percentile(graph_latencies, 95)), 2),
        "mean_recall_at_k":   round(float(np.mean(recalls)), 4),
        "speedup":            round(float(speedup), 2)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--k",         type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args.n_queries, args.k)
