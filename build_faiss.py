"""
build_faiss.py
--------------
Run this AFTER build_embeddings.py.
Builds the FAISS IVF index and saves it to disk.

Usage:
    pip install faiss-cpu
    python build_faiss.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from faiss_index import FAISSIndex

print("Loading embeddings...")
embeddings = np.load("embeddings/embeddings.npy")
print(f"Shape: {embeddings.shape}")

index = FAISSIndex(nlist=100, nprobe=10)
index.build(embeddings)
index.save("embeddings/faiss.index")
print("Done.")
