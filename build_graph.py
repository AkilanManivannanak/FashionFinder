"""
build_graph.py
--------------
Run this AFTER build_embeddings.py.
Builds the k-NN graph from saved embeddings and saves it to disk.

Usage:
    python build_graph.py --embeddings embeddings/embeddings.npy
                          --out        embeddings/knn_graph.pkl
                          --k          10
"""

import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from knn_graph import KNNGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings/embeddings.npy")
    parser.add_argument("--out",        default="embeddings/knn_graph.pkl")
    parser.add_argument("--k",          type=int, default=10)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    graph = KNNGraph(k_neighbors=args.k)
    graph.build(embeddings)
    graph.save(args.out)
    print("Done. Graph saved to", args.out)


if __name__ == "__main__":
    main()
