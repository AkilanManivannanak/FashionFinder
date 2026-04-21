# FashionFinder

Python ResNet18 FAISS k-NN Graph FastAPI Streamlit License

## What Is FashionFinder?

FashionFinder is a production-style visual search and image retrieval system that, given a query fashion product image, returns the most visually similar items — similar to how Amazon's "Find similar items" or Pinterest Visual Search works.

Built on the **Fashion Product Images Dataset (Kaggle, 44,419 products)**, the system implements and benchmarks three retrieval methods side by side:

- **Baseline** — exact brute-force cosine similarity (ground truth)
- **k-NN Graph** — approximate graph traversal via prebuilt adjacency list
- **FAISS IVF** — cluster-based approximate nearest neighbor (sub-linear)

All retrieval methods are pre-filtered by a **Hash Table** category index and a **Nested Hash Table** color index before any similarity is computed.

---

## Key Numbers

| Metric | Value | Target | Status |
|---|---|---|---|
| FAISS Median Latency | 1.79 ms | < 10 ms | 4.8x faster than baseline |
| FAISS p95 Latency | 4.04 ms | < 20 ms | Well within budget |
| Baseline Median Latency | 8.63 ms | Reference | Exact ground truth |
| Graph Median Latency | 9.37 ms | Reference | Approximate |
| FAISS Recall@10 | 0.900 | > 0.85 | 90% of baseline top-10 |
| Graph Recall@10 | 0.894 | > 0.85 | 89.4% of baseline top-10 |
| Hash Index space reduction | ~70% | > 50% | Category pre-filter |
| Products indexed | 44,419 | Full dataset | Fashion Product Images |
| Embedding dimension | 512 | ResNet18 | L2-normalized |
| k-NN Graph edges | 444,190 | k=10 per node | Precomputed |
| FAISS IVF clusters | 100 | nlist=100 | nprobe=10 at query time |
| API endpoints | 8 | FastAPI | Search, compare, benchmark |
| Benchmark queries | 80 | Reproducible | All numbers measured live |

---

## What's New (April 2026)

### 1. FAISS IVF Approximate Nearest Neighbor Index

`faiss_index.py`

Adds a third retrieval method using Facebook AI Similarity Search:

- Clusters all 44,419 embeddings into 100 Voronoi cells at build time
- At query time searches only the nearest 10 cells (nprobe=10), not all
- **1.79ms median latency** vs 8.63ms baseline: **4.8x faster**
- 90% Recall@10: finds 9 out of 10 same results as exact search

```bash
python build_faiss.py
# FAISS index built: 44419 vectors, dim=512
# FAISS index saved to embeddings/faiss.index
```

### 2. Nested Hash Table Color Index

`color_index.py`

Two-level hash table enabling combined category + color filtering:

| Filter | Structure | Result |
|---|---|---|
| Category only | Hash Table (1 level) | ~70% space reduction |
| Category + Color | Nested Hash Table (2 levels) | Further reduction |
| Example: Apparel + Navy Blue | color_index["Apparel"]["Navy Blue"] | Exact bucket |

```python
# Two-level lookup: O(1)
indices = color_index.get_indices(category="Apparel", color="Navy Blue")
```

### 3. Live Benchmark Endpoint + Charts

`api/main.py` — `/benchmark` endpoint

Runs all 3 methods over N random queries and returns latency + recall stats:

- Median / p95 / p99 latency per method
- Recall@k (graph and FAISS vs baseline ground truth)
- Rendered as live bar charts inside the Streamlit UI

```bash
curl -X POST "http://127.0.0.1:8001/benchmark?n=80&k=10"
```

### 4. Graph Visualization Endpoint

`api/main.py` — `/graph_neighbors/{product_id}`

BFS traversal of the k-NN adjacency list up to N hops, returned as node and edge data for interactive graph rendering in the UI using NetworkX and Matplotlib.

```bash
curl "http://127.0.0.1:8001/graph_neighbors/1163?hops=2"
# Returns: 53 nodes, color-coded by category
```

---

## Architecture

### Pipeline

```
44,419 Fashion Product Images (Kaggle)
        |
        v
[ResNet18 CNN Backbone]
Pretrained ImageNet weights
Final FC layer removed
Output: 512-dim per image, L2-normalized
        |
        v
[build_embeddings.py]  run once
embeddings/embeddings.npy  shape=(44419, 512)
embeddings/metadata.csv    product name, category, colour, article type
        |
        +------------------------------+
        |                              |
        v                              v
[build_graph.py]               [build_faiss.py]
k-NN Graph                     FAISS IVF Index
k=10 neighbors per node        nlist=100 clusters
444,190 total edges            87 MB index file
Saved: knn_graph.pkl           Saved: faiss.index
        |
        v
[Query Time]
     |
     v
[Hash Table]  hash_index.py
category -> list of indices, O(1) lookup, ~70% space reduction
     |
     v
[Nested Hash Table]  color_index.py
category -> color -> list of indices, second-level filter
     |
     +---------------------+---------------------+
     |                     |                     |
     v                     v                     v
[BASELINE]            [k-NN GRAPH]          [FAISS IVF]
Exact brute-force     Adjacency list        Cluster-based ANN
cosine over all       traversal             nlist=100 cells
candidates            2-hop BFS expansion   nprobe=10 at query
     |                     |                     |
     v                     v                     v
[Min-Heap]            [Min-Heap]            [Min-Heap]
heap_ranker.py        heap_ranker.py        heap_ranker.py
O(n log k) top-k      O(n log k) top-k      O(n log k) top-k
     |                     |                     |
     +---------------------+---------------------+
                           |
                           v
              FastAPI (8 endpoints, port 8001)
                           |
                           v
              Streamlit UI (3 tabs, port 8502)
              Search | Benchmark Charts | Graph Visualization
```

---

## Data Structures

| Structure | File | Role | Complexity |
|---|---|---|---|
| Hash Table | `hash_index.py` | Category to product indices | O(1) lookup |
| Nested Hash Table | `color_index.py` | Category + color to indices | O(1) lookup |
| k-NN Graph (Adjacency List) | `knn_graph.py` | Precomputed similarity graph | O(log n) traversal |
| Min-Heap / Priority Queue | `heap_ranker.py` | Top-k ranking during retrieval | O(n log k) |
| FAISS IVF Index | `faiss_index.py` | Cluster-based ANN search | Sub-linear |

---

## Benchmark Results (80 queries, k=10)

```
=======================================================
  FASHIONFINDER BENCHMARK  |  n=80  k=10
=======================================================

  Metric                    Baseline    Graph    FAISS
  ----------------------------------------------------
  Latency Median (ms)           8.63     9.37     1.79
  Latency p95 (ms)             10.75    11.70     4.04
  Latency p99 (ms)             25.91    12.78     6.27
  Recall@k                     1.000    0.894    0.900

  FAISS speedup vs baseline: 4.8x faster
  Hash index space reduction: ~70% (category pre-filter)
=======================================================
```

**Interpretation:**

- Baseline is exact (Recall 1.000) but scans all candidates in its category bucket on every query
- FAISS is **4.8x faster** than baseline with **90% recall**: finds 9 of 10 same results
- Graph traversal (k-NN adjacency list) trades a small recall loss for structure that scales better as dataset grows beyond 1M products
- Hash Table + Nested Hash Table pre-filter reduces candidates by ~70% before any method runs, benefiting all three equally

---

## Postmortem

| Issue | Root Cause | Fix | Lesson |
|---|---|---|---|
| FAISS crashes on Mac | PyTorch and FAISS both load libomp.dylib, OpenMP conflict | Set KMP_DUPLICATE_LIB_OK=TRUE at uvicorn startup | Check for OpenMP conflicts when mixing PyTorch and FAISS on Apple Silicon |
| compare all 3 times out | Three sequential API calls exceed 30s default timeout | Increase Streamlit request timeout to 120s | Set generous timeouts for multi-method endpoints |
| Wrong API served | Two main.py files in different folders; uvicorn picked StreamLens instead of FashionFinder | Run uvicorn on dedicated port 8001 per project | Always use explicit ports per project when running multiple APIs |
| Graph slower than baseline | On 44K products with hash pre-filter, brute-force over small bucket is faster than graph traversal overhead | Documented honestly; graph wins at 1M+ products | Report real numbers, do not tune to make graph look faster than it is |
| Large files warning on GitHub | embeddings.npy (86 MB) and faiss.index (87 MB) exceed GitHub 50 MB limit | Add to .gitignore; document rebuild steps in README | Always gitignore large binary artifacts and document how to regenerate them |

---

## MLOps

```
Embedding:     ResNet18 pretrained (torchvision): no training required
Build:         build_embeddings.py: one-time pass over 44,419 images (~10 min CPU)
Graph:         build_graph.py: batched matrix multiply, 25 seconds for 44K nodes
FAISS:         build_faiss.py: IVF training + add, <10 seconds
Serving:       FastAPI + Uvicorn, KMP_DUPLICATE_LIB_OK=TRUE for Mac OpenMP compat
UI:            Streamlit 3-tab interface, live benchmark charts via Matplotlib + NetworkX
Evaluation:    benchmark.py: Recall@k + latency (median / p95 / p99) over N queries
Hardware:      Apple M-series CPU: no GPU required
Dataset:       Fashion Product Images Dataset (Kaggle): 44,419 products, styles.csv
Artifacts:     embeddings/embeddings.npy (86 MB), knn_graph.pkl, faiss.index (87 MB)
```

---

## Project Structure

```
fashionfinder/
├── archive-2/
│   ├── images/                  <- 44,419 product .jpg files
│   └── styles.csv               <- product metadata
├── embeddings/
│   ├── embeddings.npy           <- (44419, 512) ResNet18 feature vectors
│   ├── metadata.csv             <- aligned product metadata
│   ├── knn_graph.pkl            <- adjacency list (k=10, 444,190 edges)
│   └── faiss.index              <- FAISS IVF index (nlist=100)
├── embedder.py                  <- ResNet18 feature extractor
├── hash_index.py                <- Hash Table: category -> indices
├── color_index.py               <- Nested Hash Table: category -> color -> indices
├── knn_graph.py                 <- k-NN Graph + adjacency list + BFS traversal
├── heap_ranker.py               <- Min-Heap top-k ranking (O(n log k))
├── faiss_index.py               <- FAISS IVF approximate nearest neighbor
├── searcher.py                  <- Unified baseline / graph / FAISS search interface
├── main.py                      <- FastAPI backend (8 endpoints)
├── ui.py                        <- Streamlit UI (Search / Benchmark / Graph tabs)
├── build_embeddings.py          <- Step 1: extract embeddings from all images
├── build_graph.py               <- Step 2: build k-NN graph from embeddings
├── build_faiss.py               <- Step 3: build FAISS IVF index
├── benchmark.py                 <- Recall@k + latency benchmark (all 3 methods)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install faiss-cpu networkx matplotlib
```

### 2. Download dataset

Download the Fashion Product Images Dataset from Kaggle:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Extract so your folder looks like:
```
fashionfinder/archive-2/images/      <- all .jpg files
fashionfinder/archive-2/styles.csv   <- metadata
```

### 3. Build embeddings (run once, ~10 minutes)

```bash
python build_embeddings.py --images_dir archive-2/images \
                           --styles_csv archive-2/styles.csv \
                           --out_dir embeddings/
# Embeddings: embeddings/embeddings.npy  shape=(44419, 512)
```

### 4. Build k-NN graph (run once, ~25 seconds)

```bash
python build_graph.py --embeddings embeddings/embeddings.npy \
                      --out embeddings/knn_graph.pkl --k 10
# Graph built: 44419 nodes, 444190 total edges
```

### 5. Build FAISS index (run once, under 10 seconds)

```bash
python build_faiss.py
# FAISS index built: 44419 vectors, dim=512
```

### 6. Start the API

```bash
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --port 8001 --workers 2 --timeout-keep-alive 120
# All components loaded. 44,419 products indexed.
```

### 7. Start the UI (new terminal)

```bash
streamlit run ui.py --server.port 8502
```

Open: http://localhost:8502

### 8. Run benchmark

```bash
python benchmark.py --n_queries 80 --k 10
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status, confirms graph + FAISS loaded |
| GET | `/categories` | All category buckets with product counts (Hash Index) |
| GET | `/colors` | All colours per category (Nested Hash Index) |
| POST | `/search/upload` | Upload image, get top-k similar products |
| POST | `/search/by_id` | Search by product ID already in dataset |
| GET | `/image/{id}` | Serve product image file |
| GET | `/graph_neighbors/{id}` | BFS neighbors for graph visualization |
| POST | `/benchmark` | Run all 3 methods over N random queries |

API docs: http://localhost:8001/docs

---

## Comparison: Baseline vs Graph vs FAISS

| Feature | Baseline | k-NN Graph | FAISS IVF |
|---|---|---|---|
| Search type | Exact | Approximate | Approximate |
| Median latency | 8.63 ms | 9.37 ms | **1.79 ms** |
| p95 latency | 10.75 ms | 11.70 ms | **4.04 ms** |
| Recall@10 | 1.000 | 0.894 | 0.900 |
| Build time | None | ~25 sec | <10 sec |
| Scales to 1M+ | No | Yes | Yes |
| Data structure | Heap scan | Adjacency list | IVF cluster index |

---

## Built By

**Akila Lourdes Miriyala Francis** | MS Artificial Intelligence, LIU Brooklyn
GitHub: AKilalours

**Akilan Manivannan** | MS Artificial Intelligence, LIU Brooklyn
GitHub: AkilanManivannanak

---

*1.79 ms FAISS · 8.63 ms Baseline · Recall@10=0.900 · 44,419 products · k-NN Graph 444,190 edges · Hash Table + Nested Color Index · FastAPI + Streamlit · Data Structures Course Project · LIU Brooklyn · April 2026*
