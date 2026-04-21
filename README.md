# FashionFinder - Visual Search & Image Retrieval System

A production-style visual search system that, given a fashion product image,
returns the most visually similar items - similar to Amazon's "Find similar items"
or Pinterest Visual Search.

Built for: Data Structures Course Final Project
Author: Akila Lourdes Miriyala Francis

---

## Data Structures Used

| Structure | File | Role |
|---|---|---|
| Hash Table | engine/hash_index.py | Index products by category, reduce search space by ~70% |
| k-NN Graph (Adjacency List) | engine/knn_graph.py | Similarity graph: each node connects to its k nearest neighbors |
| Min-Heap / Priority Queue | engine/heap_ranker.py | Track top-k results in O(n log k) without sorting full list |

---

## Architecture

```
Query Image
     |
     v
[ResNet18 Embedder]  (512-dim L2-normalized vector)
     |
     v
[Hash Table]  (category lookup -> reduces candidates ~70%)
     |
     +------------------+
     |                  |
     v                  v
[BASELINE]          [GRAPH SEARCH]
Brute-force         k-NN graph traversal
cosine over         + heap-ranked results
all candidates      2-hop expansion
     |                  |
     v                  v
[Min-Heap]          [Min-Heap]
Top-k results       Top-k results
     |                  |
     +------------------+
     |
     v
FastAPI /search  ->  Streamlit UI
```

---

## Setup

### Step 1 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 - Download dataset
Download the Fashion Product Images Dataset from Kaggle:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Extract to:
```
fashionfinder/
  data/
    fashion-dataset/
      images/       <- all .jpg files here
      styles.csv    <- metadata file here
```

### Step 3 - Build embeddings (run once)
```bash
python build_embeddings.py

# For a quick test with fewer images:
python build_embeddings.py --limit 2000
```

### Step 4 - Build k-NN graph (run once, after embeddings)
```bash
python build_graph.py
```

### Step 5 - Start the FastAPI backend
```bash
uvicorn api.main:app --reload --port 8000
```

### Step 6 - Start the Streamlit UI
```bash
streamlit run app/ui.py
```

Open: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check, confirms all components loaded |
| GET | /categories | All category buckets with product counts (Hash Index) |
| POST | /search/upload | Upload image file, get top-k similar products |
| POST | /search/by_id | Search by product ID already in dataset |
| GET | /product/{id} | Get metadata for one product |
| GET | /image/{id} | Serve product image file |
| GET | /compare/{id} | Run BOTH methods side by side + speedup |

API docs: http://localhost:8000/docs

---

## Evaluation & Benchmarking

```bash
python eval/benchmark.py --n_queries 200 --k 10
```

Output:
```
=======================================================
  FASHIONFINDER BENCHMARK  |  n=200  k=10
=======================================================

  Metric                         Baseline      Graph
  --------------------------------------------------
  Latency Median (ms)               XX.XX      XX.XX
  Latency p95 (ms)                  XX.XX      XX.XX
  Latency p99 (ms)                  XX.XX      XX.XX
  Recall@k                          1.000      0.9XX

  Speedup (graph vs baseline): X.XXx faster
=======================================================
```

---

## Project Structure

```
fashionfinder/
├── data/
│   └── fashion-dataset/
│       ├── images/              <- product images (.jpg)
│       └── styles.csv           <- product metadata
├── embeddings/
│   ├── embeddings.npy           <- (N, 512) ResNet feature vectors
│   ├── metadata.csv             <- aligned product metadata
│   └── knn_graph.pkl            <- adjacency list (k-NN graph)
├── engine/
│   ├── embedder.py              <- ResNet18 feature extractor
│   ├── hash_index.py            <- Hash table: category -> indices
│   ├── knn_graph.py             <- k-NN graph + adjacency list + traversal
│   ├── heap_ranker.py           <- Min-heap top-k ranking
│   └── searcher.py              <- Unified baseline vs graph search
├── api/
│   └── main.py                  <- FastAPI backend (7 endpoints)
├── app/
│   └── ui.py                    <- Streamlit visual demo
├── eval/
│   └── benchmark.py             <- Recall@k + latency comparison
├── build_embeddings.py          <- Step 1: extract embeddings
├── build_graph.py               <- Step 2: build k-NN graph
└── requirements.txt
```
