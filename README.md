<div align="center">

```
███████╗ █████╗ ███████╗██╗  ██╗██╗ ██████╗ ███╗   ██╗    ███████╗██╗███╗   ██╗██████╗ ███████╗██████╗
██╔════╝██╔══██╗██╔════╝██║  ██║██║██╔═══██╗████╗  ██║    ██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗
█████╗  ███████║███████╗███████║██║██║   ██║██╔██╗ ██║    █████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝
██╔══╝  ██╔══██║╚════██║██╔══██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗
██║     ██║  ██║███████║██║  ██║██║╚██████╔╝██║ ╚████║    ██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

### ⚡ Visual Search & Image Retrieval System ⚡
### *Amazon-scale similarity search · Pinterest-style visual discovery · Production-grade ML pipeline*

---

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-IVF_ANN-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-8_Endpoints-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-3_Tabs-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-44%2C419_Products-gold?style=for-the-badge)

---

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        🏆  KEY NUMBERS                                   ║
╠══════════════════════╦═══════════════╦══════════════╦════════════════════╣
║  Metric              ║  Value        ║  Target      ║  Status            ║
╠══════════════════════╬═══════════════╬══════════════╬════════════════════╣
║  FAISS Median        ║  1.79 ms      ║  < 10 ms     ║  ✅ 4.8x FASTER    ║
║  FAISS p95           ║  4.04 ms      ║  < 20 ms     ║  ✅ PASS           ║
║  Baseline Median     ║  8.63 ms      ║  Reference   ║  📊 Exact          ║
║  Graph Median        ║  9.37 ms      ║  Reference   ║  📊 Approximate    ║
║  FAISS Recall@10     ║  0.900        ║  > 0.85      ║  ✅ PASS           ║
║  Graph Recall@10     ║  0.894        ║  > 0.85      ║  ✅ PASS           ║
║  Hash Index Savings  ║  ~70%         ║  > 50%       ║  ✅ PASS           ║
║  Products Indexed    ║  44,419       ║  Full DS     ║  ✅ COMPLETE       ║
║  Embedding Dim       ║  512          ║  ResNet18    ║  ✅ L2-normalized  ║
║  k-NN Graph Edges    ║  444,190      ║  k=10/node   ║  ✅ Precomputed    ║
║  FAISS IVF Clusters  ║  100          ║  nlist=100   ║  ✅ nprobe=10      ║
║  API Endpoints       ║  8            ║  FastAPI     ║  ✅ LIVE           ║
╚══════════════════════╩═══════════════╩══════════════╩════════════════════╝
```

</div>

---

## 🔍 What Is FashionFinder?

FashionFinder is a **production-style visual search and image retrieval system** — given a query fashion product image, it returns the most visually similar items in milliseconds. The same technology powers Amazon's "Find similar items" and Pinterest's visual search.

Built on **44,419 real fashion products (Kaggle)**, the system implements, benchmarks, and visualizes **three retrieval methods** head-to-head:

| Method | Type | Median Latency | Recall@10 |
|---|---|---|---|
| 🔵 Baseline | Exact brute-force | 8.63 ms | 1.000 |
| 🟢 k-NN Graph | Approximate traversal | 9.37 ms | 0.894 |
| 🩷 FAISS IVF | Cluster-based ANN | **1.79 ms** | 0.900 |

Every candidate set is **pre-filtered by a Hash Table** (category) and a **Nested Hash Table** (color) before any similarity computation — reducing search space by ~70%.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    44,419 Fashion Product Images  (Kaggle)                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ResNet18 CNN Backbone                               │
│          Pretrained ImageNet · FC layer removed · 512-dim output            │
│                    L2-normalized for cosine = dot product                   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  build_embeddings.py  (run once ~10 min)
                                   │  embeddings/embeddings.npy  (44419, 512)
                                   ▼
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│     build_graph.py      │               │     build_faiss.py      │
│  k-NN Graph  k=10/node  │               │   FAISS IVF  nlist=100  │
│  444,190 total edges    │               │   87 MB index on disk   │
│  knn_graph.pkl          │               │   faiss.index           │
└─────────────────────────┘               └─────────────────────────┘
                                   │
                        ┌──────────┘  QUERY TIME
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hash Table  (hash_index.py)                              │
│         category -> list of indices    O(1) lookup    ~70% reduction        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Nested Hash Table  (color_index.py)                        │
│     category -> color -> list of indices    O(1) lookup    further filter   │
└──────────┬─────────────────────────┬──────────────────────────┬─────────────┘
           │                         │                          │
           ▼                         ▼                          ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────────┐
│   BASELINE      │       │   k-NN GRAPH    │       │    FAISS IVF        │
│  brute-force    │       │  adjacency list │       │  cluster-based ANN  │
│  cosine scan    │       │  2-hop BFS      │       │  nprobe=10 cells    │
│  exact results  │       │  approximate    │       │  approximate        │
└────────┬────────┘       └────────┬────────┘       └──────────┬──────────┘
         │                         │                           │
         ▼                         ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────────┐
│   Min-Heap      │       │   Min-Heap      │       │    Min-Heap         │
│ heap_ranker.py  │       │ heap_ranker.py  │       │  heap_ranker.py     │
│  O(n log k)     │       │  O(n log k)     │       │   O(n log k)        │
└────────┬────────┘       └────────┬────────┘       └──────────┬──────────┘
         └─────────────────────────┴───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  FastAPI  (8 endpoints :8001)│
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Streamlit UI  (:8502)      │
                    │  Search · Benchmark · Graph  │
                    └──────────────────────────────┘
```

---

## 🧩 Data Structures

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                          FIVE DATA STRUCTURES IN ACTION                              ║
╠═══════════════════════╦══════════════════════╦═══════════════════════════════════════╣
║  Structure            ║  File                ║  Role                                 ║
╠═══════════════════════╬══════════════════════╬═══════════════════════════════════════╣
║  Hash Table           ║  hash_index.py       ║  category -> indices  O(1) lookup     ║
║  Nested Hash Table    ║  color_index.py      ║  category -> color -> indices  O(1)   ║
║  k-NN Graph (AdjList) ║  knn_graph.py        ║  Similarity graph  O(log n) traversal ║
║  Min-Heap / PQ        ║  heap_ranker.py      ║  Top-k ranking  O(n log k)            ║
║  FAISS IVF Index      ║  faiss_index.py      ║  Cluster ANN  sub-linear search       ║
╚═══════════════════════╩══════════════════════╩═══════════════════════════════════════╝
```

### 1. Hash Table — `hash_index.py`
Maps `masterCategory` to a list of product row indices. At query time, one O(1) dictionary lookup returns only the relevant bucket, cutting the search space by ~70% before any similarity is computed.

### 2. Nested Hash Table — `color_index.py`
Two-level dictionary: `category -> color -> [indices]`. Enables combined filtering (e.g. "Apparel + Navy Blue") with a single O(1) chain lookup, reducing candidates further beyond category alone.

### 3. k-NN Graph + Adjacency List — `knn_graph.py`
Each of the 44,419 product nodes connects to its k=10 most similar neighbors (444,190 total edges), precomputed via batched einsum. At query time, BFS traversal follows edges instead of scanning all products — scalable to 1M+ items where brute-force breaks.

### 4. Min-Heap / Priority Queue — `heap_ranker.py`
During any scan or traversal, a min-heap of size k tracks the top-k best results seen so far. When a new score beats the worst item in the heap, it evicts and inserts — O(n log k) vs O(n log n) for a full sort. At k=10 and n=10,000 this is ~4x fewer operations.

### 5. FAISS IVF Index — `faiss_index.py`
Facebook AI Similarity Search clusters all 512-dim embeddings into 100 Voronoi cells at build time. At query time only the nearest 10 cells (nprobe=10) are searched — sub-linear ANN. Delivers **1.79ms median** vs 8.63ms baseline with 90% recall.

---

## 📊 Benchmark Results

```
╔═══════════════════════════════════════════════════════════════╗
║          FASHIONFINDER BENCHMARK  |  n=80 queries  k=10       ║
╠═══════════════════════╦═══════════╦═══════════╦═══════════════╣
║  Metric               ║  Baseline ║   Graph   ║     FAISS     ║
╠═══════════════════════╬═══════════╬═══════════╬═══════════════╣
║  Latency Median (ms)  ║    8.63   ║    9.37   ║  ⚡  1.79     ║
║  Latency p95   (ms)   ║   10.75   ║   11.70   ║  ⚡  4.04     ║
║  Latency p99   (ms)   ║   25.91   ║   12.78   ║  ⚡  6.27     ║
║  Recall@k             ║   1.000   ║   0.894   ║     0.900     ║
║  Search Type          ║   Exact   ║  Approx   ║    Approx     ║
║  Scales to 1M+        ║    No     ║   Yes     ║     Yes       ║
╠═══════════════════════╬═══════════╩═══════════╩═══════════════╣
║  FAISS vs Baseline    ║  4.8x FASTER  |  90% recall retained  ║
║  Hash Index Savings   ║  ~70% search space reduction          ║
╚═══════════════════════╩═══════════════════════════════════════╝
```

**Key insight:** FAISS is **4.8x faster** than baseline while finding 9 of 10 same results. The graph wins at scale (1M+ products) where brute-force becomes infeasible. Hash Table + Nested Hash Table pre-filter benefits all three methods equally.

---

## 🔥 What's New (April 2026)

### ⚡ FAISS IVF Approximate Nearest Neighbor Index — `faiss_index.py`
- Clusters 44,419 embeddings into 100 Voronoi cells at build time
- Searches only nearest 10 cells at query time (nprobe=10)
- **1.79ms median** vs 8.63ms baseline — **4.8x faster**
- 90% Recall@10 on 80-query live benchmark

### 🎨 Nested Hash Table Color Index — `color_index.py`
- Two-level map: `category -> color -> [indices]`
- 47 unique colours across 7 categories
- O(1) combined filtering — "Apparel + Navy Blue" in one lookup

### 📈 Live Benchmark Endpoint + Charts — `/benchmark`
- Runs all 3 methods over N random queries server-side
- Returns median / p95 / p99 latency + Recall@k per method
- Rendered as live Matplotlib bar charts in the Streamlit UI

### 🕸️ Graph Visualization — `/graph_neighbors/{id}`
- BFS traversal of k-NN adjacency list up to 3 hops
- 53 nodes at 2 hops for product 1163
- Color-coded by category, rendered via NetworkX + Matplotlib
- Connected products table with similarity scores and depth

---

## 🚨 Postmortem

```
╔══════════════════════════╦═══════════════════════════════╦═══════════════════════════════════╦════════════════════════════════╗
║  Issue                   ║  Root Cause                   ║  Fix                              ║  Lesson                        ║
╠══════════════════════════╬═══════════════════════════════╬═══════════════════════════════════╬════════════════════════════════╣
║  FAISS crashes on Mac    ║  PyTorch + FAISS both load    ║  KMP_DUPLICATE_LIB_OK=TRUE        ║  Check OpenMP conflicts on     ║
║                          ║  libomp.dylib, OpenMP clash   ║  at uvicorn startup               ║  Apple Silicon                 ║
╠══════════════════════════╬═══════════════════════════════╬═══════════════════════════════════╬════════════════════════════════╣
║  Compare all 3 times out ║  3 sequential API calls       ║  Increase Streamlit timeout       ║  Set generous timeouts for     ║
║                          ║  exceed 30s default           ║  to 120s                          ║  multi-method endpoints        ║
╠══════════════════════════╬═══════════════════════════════╬═══════════════════════════════════╬════════════════════════════════╣
║  Wrong API served        ║  Two main.py in different     ║  Dedicated port 8001 per          ║  Always use explicit ports     ║
║                          ║  folders, uvicorn picked      ║  project                          ║  per project                   ║
║                          ║  StreamLens instead           ║                                   ║                                ║
╠══════════════════════════╬═══════════════════════════════╬═══════════════════════════════════╬════════════════════════════════╣
║  Graph slower than       ║  44K products with hash       ║  Documented honestly — graph      ║  Report real numbers, do not   ║
║  baseline                ║  pre-filter too small for     ║  wins at 1M+ products             ║  tune to look better           ║
║                          ║  graph overhead to pay off    ║                                   ║                                ║
╠══════════════════════════╬═══════════════════════════════╬═══════════════════════════════════╬════════════════════════════════╣
║  GitHub large file warn  ║  embeddings.npy 86MB and      ║  Add to .gitignore, document      ║  Always gitignore large        ║
║                          ║  faiss.index 87MB exceed      ║  rebuild steps in README          ║  binary artifacts              ║
║                          ║  GitHub 50MB limit            ║                                   ║                                ║
╚══════════════════════════╩═══════════════════════════════╩═══════════════════════════════════╩════════════════════════════════╝
```

---

## ⚙️ MLOps

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOPS SUMMARY                            │
├─────────────────┬───────────────────────────────────────────────┤
│  Embedding      │  ResNet18 pretrained (torchvision)            │
│                 │  No training required                         │
├─────────────────┼───────────────────────────────────────────────┤
│  Build          │  build_embeddings.py                          │
│                 │  One-time pass over 44,419 images (~10 min)   │
├─────────────────┼───────────────────────────────────────────────┤
│  Graph          │  build_graph.py                               │
│                 │  Batched matrix multiply, 25 sec for 44K nodes│
├─────────────────┼───────────────────────────────────────────────┤
│  FAISS          │  build_faiss.py                               │
│                 │  IVF training + add, under 10 seconds         │
├─────────────────┼───────────────────────────────────────────────┤
│  Serving        │  FastAPI + Uvicorn                            │
│                 │  KMP_DUPLICATE_LIB_OK=TRUE (Mac OpenMP fix)   │
├─────────────────┼───────────────────────────────────────────────┤
│  UI             │  Streamlit 3-tab interface                    │
│                 │  Matplotlib charts + NetworkX graph viz       │
├─────────────────┼───────────────────────────────────────────────┤
│  Evaluation     │  benchmark.py                                 │
│                 │  Recall@k + latency median/p95/p99            │
├─────────────────┼───────────────────────────────────────────────┤
│  Hardware       │  Apple M-series CPU, no GPU required          │
├─────────────────┼───────────────────────────────────────────────┤
│  Dataset        │  Fashion Product Images Dataset (Kaggle)      │
│                 │  44,419 products, styles.csv metadata         │
├─────────────────┼───────────────────────────────────────────────┤
│  Artifacts      │  embeddings.npy 86MB                          │
│                 │  knn_graph.pkl, faiss.index 87MB              │
└─────────────────┴───────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
fashionfinder/
│
├── 📦 Data
│   ├── archive-2/images/            ← 44,419 product .jpg files
│   └── archive-2/styles.csv         ← product metadata
│
├── 🧠 Embeddings (generated)
│   ├── embeddings/embeddings.npy    ← (44419, 512) ResNet18 vectors
│   ├── embeddings/metadata.csv      ← aligned product metadata
│   ├── embeddings/knn_graph.pkl     ← adjacency list (444,190 edges)
│   └── embeddings/faiss.index       ← FAISS IVF index (nlist=100)
│
├── ⚙️ Engine (Data Structures)
│   ├── embedder.py                  ← ResNet18 feature extractor
│   ├── hash_index.py                ← Hash Table: category -> indices
│   ├── color_index.py               ← Nested Hash Table: cat -> color -> idx
│   ├── knn_graph.py                 ← k-NN Graph + adjacency list + BFS
│   ├── heap_ranker.py               ← Min-Heap top-k ranking O(n log k)
│   ├── faiss_index.py               ← FAISS IVF approximate nearest neighbor
│   └── searcher.py                  ← Unified baseline/graph/FAISS interface
│
├── 🔨 Build Scripts
│   ├── build_embeddings.py          ← Step 1: extract all embeddings
│   ├── build_graph.py               ← Step 2: build k-NN graph
│   └── build_faiss.py               ← Step 3: build FAISS IVF index
│
├── 🚀 Serving
│   ├── main.py                      ← FastAPI backend (8 endpoints)
│   └── ui.py                        ← Streamlit UI (Search/Benchmark/Graph)
│
├── 📊 Evaluation
│   └── benchmark.py                 ← Recall@k + latency all 3 methods
│
└── 📋 requirements.txt
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
pip install faiss-cpu networkx matplotlib
```

### 2. Download Dataset
```
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
```
Extract to `fashionfinder/archive-2/`

### 3. Build Embeddings (once, ~10 min)
```bash
python build_embeddings.py --images_dir archive-2/images \
                           --styles_csv archive-2/styles.csv \
                           --out_dir embeddings/
# ✅ Embeddings: shape=(44419, 512)
```

### 4. Build k-NN Graph (once, ~25 sec)
```bash
python build_graph.py --embeddings embeddings/embeddings.npy \
                      --out embeddings/knn_graph.pkl --k 10
# ✅ Graph: 44419 nodes, 444190 edges
```

### 5. Build FAISS Index (once, <10 sec)
```bash
python build_faiss.py
# ✅ FAISS: 44419 vectors indexed
```

### 6. Start API
```bash
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --port 8001 --workers 2 --timeout-keep-alive 120
# ✅ All components loaded. 44,419 products indexed.
```

### 7. Start UI (new terminal)
```bash
streamlit run ui.py --server.port 8502
# ✅ Open: http://localhost:8502
```

### 8. Run Benchmark
```bash
python benchmark.py --n_queries 80 --k 10
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status, graph + FAISS confirmation |
| `GET` | `/categories` | Hash Table buckets with product counts |
| `GET` | `/colors` | Nested Hash Index colours per category |
| `POST` | `/search/upload` | Upload image, get top-k similar products |
| `POST` | `/search/by_id` | Search by product ID in dataset |
| `GET` | `/image/{id}` | Serve product image file |
| `GET` | `/graph_neighbors/{id}` | BFS graph traversal for visualization |
| `POST` | `/benchmark` | Live 3-method benchmark over N queries |

> API docs: **http://localhost:8001/docs**

---

## 👥 Team Contributions

```
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              INDIVIDUAL CONTRIBUTIONS                                          ║
╠══════════════════════════════════════╦═════════════════════════════════════════════════════════╣
║  Akilan Manivannan                   ║  Akila Lourdes Miriyala Francis                         ║
║  GitHub: AkilanManivannanak          ║  GitHub: AKilalours                                     ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  Data Pipeline                       ║  Data Pipeline                                          ║
║  • Kaggle dataset acquisition        ║  • styles.csv metadata parsing + cleaning               ║
║  • Image validation and filtering    ║  • Product ID to filepath index mapping                 ║
║  • Dataset split strategy            ║  • Embedding pipeline orchestration                     ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  Data Structures                     ║  Data Structures                                        ║
║  • k-NN Graph design and build       ║  • Hash Table (hash_index.py)                           ║
║  • knn_graph.py adjacency list       ║  • Nested Hash Table (color_index.py)                   ║
║  • BFS traversal implementation      ║  • Min-Heap ranker (heap_ranker.py)                     ║
║  • Graph save/load (pickle)          ║  • Searcher unified interface (searcher.py)             ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  ML / Retrieval                      ║  ML / Retrieval                                         ║
║  • ResNet18 embedder (embedder.py)   ║  • FAISS IVF index (faiss_index.py)                     ║
║  • build_embeddings.py pipeline      ║  • build_faiss.py pipeline                              ║
║  • build_graph.py pipeline           ║  • Baseline brute-force cosine search                   ║
║  • Graph search method               ║  • L2 normalization strategy                            ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  Backend / API                       ║  Backend / API                                          ║
║  • /search/by_id endpoint            ║  • /search/upload endpoint                              ║
║  • /graph_neighbors BFS endpoint     ║  • /benchmark endpoint                                  ║
║  • /image file serving               ║  • /categories + /colors endpoints                      ║
║  • FastAPI routing + error handling  ║  • CORS + middleware setup                              ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  UI / Frontend                       ║  UI / Frontend                                          ║
║  • Graph Visualization tab           ║  • Search tab (upload + product ID)                     ║
║  • NetworkX + Matplotlib graph       ║  • Benchmark Charts tab                                 ║
║  • Connected products table          ║  • Live latency + recall bar charts                     ║
║  • Sidebar system stats panel        ║  • Product cards with score progress bars               ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  Evaluation                          ║  Evaluation                                             ║
║  • benchmark.py design               ║  • Recall@k metric implementation                       ║
║  • Latency measurement (p50/p95/p99) ║  • Results CSV export                                   ║
║  • Multi-method comparison logic     ║  • Interpretation + documentation                       ║
╠══════════════════════════════════════╬═════════════════════════════════════════════════════════╣
║  MLOps / DevOps                      ║  MLOps / DevOps                                         ║
║  • GitHub repo setup (AkilanManiv..) ║  • GitHub repo setup (AKilalours)                       ║
║  • SSH key configuration             ║  • SSH key push from her Mac                            ║
║  • Port isolation (8001 / 8502)      ║  • KMP_DUPLICATE_LIB_OK fix discovery                   ║
║  • Postmortem documentation          ║  • README + submission docx authoring                   ║
╚══════════════════════════════════════╩═════════════════════════════════════════════════════════╝
```

---

<div align="center">

```
⚡ 1.79 ms FAISS  ·  8.63 ms Baseline  ·  Recall@10 = 0.900
44,419 Products  ·  444,190 Graph Edges  ·  512-dim ResNet18 Embeddings
Hash Table + Nested Color Index + k-NN Graph + Min-Heap + FAISS IVF
FastAPI · Streamlit · NetworkX · Matplotlib · PyTorch · Apple Silicon
```

**Data Structures Course Project · LIU Brooklyn · April 2026**

**Akila Lourdes Miriyala Francis** · **Akilan Manivannan**

</div>
