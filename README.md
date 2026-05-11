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
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-12_Pages-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-RAG_Assistant-8A2BE2?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-44%2C419_Products-gold?style=for-the-badge)

**Built by: Akilan Manivannan (100863473) · Akila Lourdes Miriyala Francis (100863383)**

**CS 631 — Algorithms & Data Structures · Prof. Dr. Abla Bedoui · LIU Brooklyn · Spring 2026**

🚀 **[Live Demo - Full Streamlit](https://ruse-catnap-oblong.ngrok-free.dev)** (keep Mac on) &nbsp;·&nbsp; 🎯 **[Deployed Demo](https://huggingface.co/spaces/Akilanak/FashionFinder)** &nbsp;&nbsp;|&nbsp;&nbsp; 📦 **[Source Code](https://github.com/AkilanManivannanak/FashionFinder)**

</div>

---

## 🌐 Live Demo & Links

| | Link |
|--|------|
| 🚀 **Full Demo (Streamlit)** | https://ruse-catnap-oblong.ngrok-free.dev |
| 🎯 **Deployed Demo (Gradio)** | https://huggingface.co/spaces/Akilanak/FashionFinder |
| 📦 **Source Code** | https://github.com/AkilanManivannanak/FashionFinder |
| 📖 **API Docs** | http://localhost:8001/docs (local) |

---

## 📊 Key Performance Numbers

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🏆  BENCHMARK RESULTS                               ║
╠══════════════════════╦═══════════════╦══════════════╦════════════════════════╣
║  Metric              ║  Value        ║  Target      ║  Status                ║
╠══════════════════════╬═══════════════╬══════════════╬════════════════════════╣
║  FAISS Median        ║  1.79 ms      ║  < 10 ms     ║  ✅ 4.8x FASTER        ║
║  FAISS p95           ║  4.04 ms      ║  < 20 ms     ║  ✅ PASS               ║
║  FAISS p99           ║  6.27 ms      ║  < 30 ms     ║  ✅ PASS               ║
║  Baseline Median     ║  8.63 ms      ║  Reference   ║  📊 Exact              ║
║  Graph Median        ║  9.37 ms      ║  Reference   ║  📊 Approximate        ║
║  FAISS Recall@10     ║  0.900        ║  > 0.85      ║  ✅ PASS               ║
║  Graph Recall@10     ║  0.894        ║  > 0.85      ║  ✅ PASS               ║
║  Hash Index Savings  ║  ~70%         ║  > 50%       ║  ✅ PASS               ║
║  Products Indexed    ║  44,419       ║  Full DS     ║  ✅ COMPLETE           ║
║  Brands Indexed      ║  467          ║  Full DS     ║  ✅ COMPLETE           ║
║  Colors Indexed      ║  47           ║  Full DS     ║  ✅ COMPLETE           ║
║  Embedding Dim       ║  512          ║  ResNet18    ║  ✅ L2-normalized      ║
║  k-NN Graph Edges    ║  444,190      ║  k=10/node   ║  ✅ Precomputed        ║
║  FAISS IVF Clusters  ║  100          ║  nlist=100   ║  ✅ nprobe=10          ║
╚══════════════════════╩═══════════════╩══════════════╩════════════════════════╝
```

---

## 🔍 What Is FashionFinder?

FashionFinder is a **production-grade visual search and image retrieval system** built from first principles for CS 631. Given any fashion product image, it finds the most visually similar items in milliseconds across 44,419 real products.

The same core technology powers:
- **Amazon** → "Find similar items"
- **Pinterest** → Visual Lens search
- **Google Lens** → Product identification

But FashionFinder goes far beyond all of these. It is the **only fashion search system** that:
- Shows you **why** two items match (similarity scores + GradCAM heatmaps)
- Lets you **compare three retrieval algorithms** head-to-head in real time
- Provides **style transfer** — same cut, different color
- Includes a **RAG-powered AI assistant** grounded in real product data
- Visualizes the **embedding space as a graph** you can explore

### Three Retrieval Methods, Head-to-Head

| Method | Type | Median Latency | Recall@10 | Scales to 1M+ |
|--------|------|---------------|-----------|----------------|
| 🔵 Baseline | Exact brute-force cosine | 8.63 ms | 1.000 | ❌ No |
| 🟢 k-NN Graph | Approximate BFS traversal | 9.37 ms | 0.894 | ✅ Yes |
| 🩷 FAISS IVF | Cluster-based ANN | **1.79 ms** | 0.900 | ✅ Yes |

Every candidate set is pre-filtered by a **Hash Table** (category) and **Nested Hash Table** (color) before any similarity computation — reducing the search space by ~70%.

---

## 🆚 FashionFinder vs. Google Lens, Amazon & Pinterest

This comparison shows exactly where FashionFinder differentiates itself from the world's most advanced commercial visual search systems:

| Feature | Google Lens | Amazon Visual | Pinterest Lens | **FashionFinder** |
|---------|:-----------:|:-------------:|:--------------:|:-----------------:|
| Visual similarity search | ✅ | ✅ | ✅ | ✅ |
| Shows similarity scores | ❌ | ❌ | ❌ | ✅ **Explainable AI** |
| Explains why items match | ❌ | ❌ | ❌ | ✅ **GradCAM heatmaps** |
| Compare 3 retrieval algorithms | ❌ | ❌ | ❌ | ✅ **Live benchmark** |
| Real-time latency benchmark | ❌ | ❌ | ❌ | ✅ **p50/p95/p99** |
| Style Transfer (same style, new color) | ❌ | ❌ | ❌ | ✅ **Unique feature** |
| Multi-image fusion search | ❌ | ❌ | ❌ | ✅ **Blend 2-4 styles** |
| 467-brand cross-comparison | ❌ | ❌ | ❌ | ✅ **Brand Explorer** |
| MMR diversity reranking | ❌ | ❌ | ❌ | ✅ **Less duplicates** |
| AI Fashion Assistant (RAG) | ❌ | ❌ | ❌ | ✅ **Claude-powered** |
| k-NN Graph visualization | ❌ | ❌ | ❌ | ✅ **Graph explorer** |
| Outfit completion | ❌ | ❌ | ❌ | ✅ **Complete the look** |
| Style modifier search (CLIP) | ❌ | ❌ | ❌ | ✅ **Formal/casual/sporty** |
| Real-time product ingestion | ❌ | ❌ | ❌ | ✅ **Live index updates** |
| Trending products tracker | ❌ | ❌ | ❌ | ✅ **Real-time trends** |
| 100% open source | ❌ | ❌ | ❌ | ✅ **Full source code** |

### Why This Matters Academically

Commercial systems are black boxes. FashionFinder is built from first principles so every decision is visible and measurable:

- **You can see exactly why** item A is similar to item B (cosine score, GradCAM)
- **You can compare** three fundamentally different algorithms on the same query
- **You can measure** the speed vs. accuracy trade-off in real time
- **You can visualize** the graph topology of the embedding space
- **You can run** the benchmark yourself and verify every number in this README

---

## 🏗️ System Architecture

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
         ┌─────────────────────────┼────────────────────────┐
         │                         │                        │
         ▼                         ▼                        ▼
┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│  build_graph.py │   │   build_faiss.py    │   │  hash_index.py       │
│  k-NN  k=10     │   │  FAISS IVF nlist=100│   │  color_index.py      │
│  444,190 edges  │   │  87 MB index        │   │  brand_index.py      │
└────────┬────────┘   └──────────┬──────────┘   └──────────┬───────────┘
         │                       │                          │
         └───────────────────────┴──────────────────────────┘
                                 │  QUERY TIME
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pre-Filter Stage (Hash Tables)                       │
│   Hash Table (cat)    → ~70% candidate reduction   O(1)                │
│   Nested Hash (color) → further reduction          O(1)                │
│   Brand Index         → exact brand matching       O(1)                │
└──────────┬─────────────────────┬──────────────────────────┬────────────┘
           │                     │                          │
           ▼                     ▼                          ▼
┌─────────────────┐   ┌─────────────────┐       ┌────────────────────┐
│   BASELINE      │   │   k-NN GRAPH    │       │    FAISS IVF       │
│  brute-force    │   │  BFS traversal  │       │  cluster ANN       │
│  cosine scan    │   │  2-hop search   │       │  nprobe=10 cells   │
│  exact results  │   │  approximate    │       │  approximate       │
└────────┬────────┘   └────────┬────────┘       └──────────┬─────────┘
         └────────────────────┴────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Min-Heap Ranker  O(n log k)  │
                │  MMR Diversity Reranker  O(k²)│
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  FastAPI REST API  (:8001)    │
                │  20+ endpoints               │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │  Streamlit UI  (:8502)        │
                │  12 pages · All features      │
                └───────────────────────────────┘
```

---

## 🧩 Data Structures — Deep Dive

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                          SEVEN DATA STRUCTURES IN ACTION                             ║
╠═══════════════════════╦══════════════════════╦═══════════════════════════════════════╣
║  Structure            ║  File                ║  Role & Complexity                    ║
╠═══════════════════════╬══════════════════════╬═══════════════════════════════════════╣
║  Hash Table           ║  hash_index.py       ║  category → indices  O(1) lookup      ║
║  Nested Hash Table    ║  color_index.py      ║  category → color → indices  O(1)    ║
║  Brand Hash Index     ║  brand_index.py      ║  brand → category → indices  O(1)    ║
║  k-NN Graph (AdjList) ║  knn_graph.py        ║  Similarity graph  BFS  O(log n)      ║
║  Min-Heap / PQ        ║  heap_ranker.py      ║  Top-k ranking  O(n log k)            ║
║  FAISS IVF Index      ║  faiss_index.py      ║  Cluster ANN  sub-linear search       ║
║  MMR Reranker         ║  mmr_reranker.py     ║  Diversity reranking  O(k²)           ║
╚═══════════════════════╩══════════════════════╩═══════════════════════════════════════╝
```

### 1. Hash Table — `hash_index.py`

Maps `masterCategory` to a list of product row indices. At query time, one O(1) dictionary lookup returns only the relevant bucket, cutting the search space by ~70% before any expensive similarity computation.

```python
hash_index["Apparel"]  # → [0, 1, 4, 7, ...] — 21,392 indices
hash_index["Footwear"] # → [2, 3, 8, ...]     — 9,219 indices
# Without filter: scan 44,419 products
# With filter:    scan only ~21,392 (52% reduction for Apparel queries)
```

**Space complexity:** O(N) where N = number of products. **Time complexity:** O(1) lookup, O(k) build where k = number of categories.

### 2. Nested Hash Table — `color_index.py`

Two-level dictionary: `category → color → [indices]`. Enables combined filtering in a single O(1) chain lookup. 47 unique colors across 7 categories.

```python
color_index["Apparel"]["Navy Blue"]  # → ~800 indices (from 21,392)
# 96% additional reduction beyond category filter
```

**Design decision:** We store indices (not embeddings) to minimize memory. The embedding matrix is accessed separately by index, keeping the hash tables lightweight (< 5 MB total).

### 3. Brand Hash Index — `brand_index.py`

Three-level nested hash: `brand → category → [indices]`. Maps all 467 brands to their products, enabling cross-brand comparison queries at O(1) per brand.

```python
brand_index["Nike"]["Footwear"]   # → Nike shoes only
brand_index["Adidas"]["Apparel"]  # → Adidas clothing only
brand_index["Puma"]["Footwear"]   # → Puma shoes only
# Brand Compare: fetch 4 brands in 4 O(1) lookups, rank each
```

Powers the **Brand Comparison** feature — find how Nike, Adidas, Puma, and Reebok each style the same product type.

### 4. k-NN Graph + Adjacency List — `knn_graph.py`

Each of the 44,419 product nodes connects to its k=10 most similar neighbors (444,190 total edges), precomputed via batched einsum matrix multiply at build time.

```
Product 1163 (Cricket Jersey)
    ├── Neighbor 1164 (sim=0.97) → Nike Blue India Jersey
    ├── Neighbor 3313 (sim=0.95) → Mumbai Indians Jersey
    ├── Neighbor 13891 (sim=0.93) → Nike Team India Jersey
    └── ... (10 neighbors total, each with 10 more → ~100 2-hop candidates)
```

At query time, BFS traversal to depth 2 gives ~100 highly relevant candidates without scanning all 44,419 products. Scales to 1M+ products where brute force (O(N)) becomes infeasible — graph search stays O(log N).

**Build time:** ~25 seconds using batched einsum (20 batches × 2,221 products each). **Storage:** knn_graph.pkl ~87 MB.

### 5. Min-Heap / Priority Queue — `heap_ranker.py`

A fixed-size min-heap of capacity k tracks the top-k results seen during any scan. When a new score exceeds the heap minimum, the minimum is evicted and the new score inserted.

```
Insertion: O(log k)
Check minimum: O(1)
Full sort equivalent: O(n log n)
Heap-based top-k: O(n log k)

At k=10, n=10,000: ~4x fewer operations than full sort
```

The heap invariant: the root is always the **worst** of the top-k seen so far — making the "is this worth inserting?" check O(1).

### 6. FAISS IVF Index — `faiss_index.py`

Facebook AI Similarity Search clusters all 512-dim L2-normalized embeddings into 100 Voronoi cells (nlist=100) at build time using k-means. At query time, only the nearest nprobe=10 cells are searched.

```
Build:   < 10 seconds — k-means + quantize 44,419 vectors
Index:   ~87 MB on disk
Query:   nprobe=10 → search ~4,440 products (10% of total)
Latency: 1.79 ms median vs 8.63 ms brute-force (4.8x speedup)
Recall:  0.900 — finds 9 of 10 same results as exact search
```

**Why IVF (Inverted File Index)?** Each cluster contains ~444 products on average. Searching 10 clusters = ~4,440 distance computations instead of 44,419 — a 10x reduction with only 10% recall loss.

### 7. MMR Diversity Reranker — `mmr_reranker.py`

Maximal Marginal Relevance reranks the top-k results to balance relevance with diversity. Without MMR, searching "blue Nike jersey" might return 10 nearly identical jerseys.

```
MMR score = λ × relevance(d, q) − (1−λ) × max_sim(d, already_selected)

At λ=0.6:
  60% weight → relevance to query
  40% penalty → similarity to already-selected results
```

MMR selects greedily: at each step, pick the document that maximizes the MMR score. **Complexity: O(k²)** — negligible for k≤20. Result: diverse top-k that covers more of the visual space.

---

## 🌟 Feature Modules

### 🔍 Visual Search
Upload any fashion image — ResNet18 extracts a 512-dim embedding and finds the most visually similar products. Supports:
- **Image upload** — drag & drop from your computer
- **Image URL** — paste any public image URL
- **Product ID** — enter any ID from the 44,419 dataset
- **Multi-image fusion** — upload 2-4 images, blend their embeddings into one query

### 🎨 Style Transfer Search — `style_transfer_search.py`
**Unique to FashionFinder — not in Google Lens or Pinterest.**

Find the same style in a different color:
1. Embed query with ResNet18
2. Filter candidates by target color (Nested Hash Table O(1))
3. Rank by cosine similarity within the color-filtered set
4. Return: same cut/style, new color

### ✨ CLIP Style Modifier — `clip_search.py`
Upload an image + pick a style modifier ("formal", "casual", "sporty", "elegant", "vintage", "party", "minimal"). CLIP fuses the visual embedding with a text embedding to bias results toward the specified style.

```python
# CLIP fusion: 70% image, 30% text style modifier
query_vec = 0.7 × image_embedding + 0.3 × text_embedding
```

### 🏷️ Brand Cross-Comparison
Compare how Nike, Adidas, Puma, and Reebok each style the same product type side-by-side. Uses the Brand Hash Index (O(1) per brand) to fetch each brand's candidates, then ranks by cosine similarity to the query.

### 👗 Outfit Completion — `searcher.py`
Given a garment, suggest complementary items. Shoes for a dress. A jacket for trousers. Category-aware similarity search finds items that visually harmonize across different product types.

### 🧬 Visual DNA (GradCAM) — `visual_dna.py`
Shows **which pixels** drove the ResNet18 similarity match using Gradient-weighted Class Activation Mapping. Overlays a heatmap on both the query and result image — highlighting collars, patterns, colors, and textures that caused the match.

### 🕸️ k-NN Graph Visualizer
Interactive NetworkX visualization of BFS traversal from any product — up to 3 hops. Nodes are color-coded by category; edge weights reflect cosine similarity. Reveals the actual topology of the embedding space.

### 📊 Live Benchmark
Run a real-time benchmark (not pre-computed) over N random queries:
- Latency: median, p95, p99 for all three methods
- Recall@10 for graph and FAISS vs. exact baseline
- Live Matplotlib bar charts rendered in the Streamlit UI

### 🤖 AI Fashion Assistant (RAG) — `chatbot.py`
Powered by Anthropic's Claude with Retrieval-Augmented Generation:

```
Step 1 — Retrieve: FAISS finds top-20 relevant products from 44,419
Step 2 — Augment:  Product metadata (name, brand, color, type) → context
Step 3 — Generate: Claude writes advice grounded in real catalog data
```

The assistant **never hallucinates** product names — every recommendation cites a real product ID from the dataset. Ask: "What should I wear to a wedding?", "Build me a gym outfit", "Show me red Nike shoes".

### ➕ Real-Time Product Ingestion — `realtime_index.py`
Add new products to the live index without rebuilding from scratch. A `RealTimeIndex` buffer accumulates new products and merges into the main FAISS index on flush. Demonstrates production catalog update patterns.

### 🔥 Trending Products — `trend_tracker.py`
Tracks which product IDs are searched most frequently in the last N hours using a time-weighted frequency dictionary. Updates in real time with every search query.

---

## 📁 Project Structure

```
fashionfinder/
│
├── 📦 Data
│   ├── archive-2/images/            ← 44,419 product .jpg files
│   └── archive-2/styles.csv         ← product metadata (name, brand, color, category)
│
├── 🧠 Embeddings (generated — not in git, too large)
│   ├── embeddings/embeddings.npy    ← (44419, 512) float32 — 86 MB
│   ├── embeddings/metadata.csv      ← aligned product metadata
│   ├── embeddings/knn_graph.pkl     ← adjacency list (444,190 edges) — 87 MB
│   └── embeddings/faiss.index       ← FAISS IVF index (nlist=100) — 87 MB
│
├── ⚙️ Data Structures Engine
│   ├── embedder.py                  ← ResNet18 512-dim feature extractor
│   ├── hash_index.py                ← Hash Table: category → indices
│   ├── color_index.py               ← Nested Hash: category → color → indices
│   ├── brand_index.py               ← Brand Hash: brand → category → indices
│   ├── knn_graph.py                 ← k-NN Graph + adjacency list + BFS
│   ├── heap_ranker.py               ← Min-Heap top-k  O(n log k)
│   ├── faiss_index.py               ← FAISS IVF approximate nearest neighbor
│   ├── mmr_reranker.py              ← MMR diversity reranking  O(k²)
│   └── searcher.py                  ← Unified baseline / graph / FAISS interface
│
├── 🔨 Build Scripts (run once to generate embeddings)
│   ├── build_embeddings.py          ← Step 1: ResNet18 forward pass all images
│   ├── build_graph.py               ← Step 2: batched einsum k-NN graph
│   └── build_faiss.py               ← Step 3: k-means + IVF index
│
├── 🌟 Feature Modules
│   ├── style_transfer_search.py     ← Same style, new color
│   ├── clip_search.py               ← CLIP text+image fusion search
│   ├── visual_dna.py                ← GradCAM explainability heatmaps
│   ├── explainer.py                 ← Similarity score explanations
│   ├── fashion_timeline.py          ← Year-by-year trend analysis
│   ├── trend_tracker.py             ← Real-time trending products
│   ├── realtime_index.py            ← Live product ingestion buffer
│   └── chatbot.py                   ← RAG AI assistant (Anthropic Claude)
│
├── 🚀 Serving
│   ├── main.py                      ← FastAPI backend (20+ REST endpoints)
│   └── ui.py                        ← Streamlit UI (12 pages)
│
├── 📊 Evaluation
│   └── benchmark.py                 ← Recall@k + p50/p95/p99 all 3 methods
│
└── 📋 requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status — graph, FAISS, CLIP confirmation |
| `GET` | `/categories` | Hash Table buckets with product counts |
| `GET` | `/colors` | Nested Hash Index: colors per category |
| `GET` | `/brands` | Brand Hash Index: all 467 brands |
| `POST` | `/search/upload` | Upload image → top-k similar products |
| `POST` | `/search/by_id` | Search by product ID |
| `POST` | `/search/url` | Search by image URL |
| `POST` | `/clip_search` | CLIP text+image fusion search |
| `GET` | `/style_transfer/{id}` | Same style, new color |
| `GET` | `/brand_compare/{id}` | Cross-brand comparison |
| `POST` | `/outfit/complete` | Outfit completion suggestions |
| `GET` | `/image/{id}` | Serve product image file |
| `GET` | `/graph_neighbors/{id}` | BFS graph traversal + NetworkX data |
| `POST` | `/benchmark` | Live 3-method benchmark over N queries |
| `GET` | `/trending` | Trending products by time window |
| `GET` | `/search_history` | Recent search log |
| `POST` | `/realtime/add` | Add product to live index buffer |
| `POST` | `/chat` | RAG AI assistant query |
| `GET` | `/visual_dna/{id}` | GradCAM heatmap PNG |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 📊 Benchmark Results

```
╔═══════════════════════════════════════════════════════════════╗
║       FASHIONFINDER BENCHMARK  |  n=80 queries  k=10          ║
╠═══════════════════════╦═══════════╦═══════════╦═══════════════╣
║  Metric               ║  Baseline ║   Graph   ║     FAISS     ║
╠═══════════════════════╬═══════════╬═══════════╬═══════════════╣
║  Latency Median (ms)  ║    8.63   ║    9.37   ║  ⚡  1.79     ║
║  Latency p95   (ms)   ║   10.75   ║   11.70   ║  ⚡  4.04     ║
║  Latency p99   (ms)   ║   25.91   ║   12.78   ║  ⚡  6.27     ║
║  Recall@10            ║   1.000   ║   0.894   ║     0.900     ║
║  Search Type          ║   Exact   ║  Approx   ║    Approx     ║
║  Scales to 1M+        ║    No     ║   Yes     ║     Yes       ║
╠═══════════════════════╬═══════════╩═══════════╩═══════════════╣
║  FAISS vs Baseline    ║  4.8x FASTER  |  90% recall retained  ║
║  Hash Index Savings   ║  ~70% search space reduction          ║
╚═══════════════════════╩═══════════════════════════════════════╝
```

**Key insight:** FAISS is **4.8x faster** than exact search while finding 9 of 10 same results. The k-NN graph wins at 1M+ products where O(N) brute-force becomes infeasible. Hash Table pre-filtering benefits all three methods equally.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Optional: CLIP for style modifier search
pip install git+https://github.com/openai/CLIP.git
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
# Output: embeddings/embeddings.npy  shape=(44419, 512)
#         embeddings/metadata.csv
```

### 4. Build k-NN Graph (once, ~25 sec)
```bash
python build_graph.py --embeddings embeddings/embeddings.npy \
                      --out embeddings/knn_graph.pkl --k 10
# Output: 44,419 nodes, 444,190 edges
```

### 5. Build FAISS Index (once, < 10 sec)
```bash
python build_faiss.py
# Output: embeddings/faiss.index  (nlist=100, nprobe=10)
```

### 6. Start API Backend
```bash
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --port 8001 --workers 2
# Ready: 44,419 products | 467 brands | CLIP=ON
# API docs: http://localhost:8001/docs
```

### 7. Start Streamlit UI (new terminal)
```bash
streamlit run ui.py --server.port 8502
# Open: http://localhost:8502
```

### 8. Run Benchmark
```bash
python benchmark.py --n_queries 80 --k 10
# Outputs: latency p50/p95/p99 + Recall@10 for all 3 methods
```

---

## ⚙️ MLOps Summary

| Component | Details |
|-----------|---------|
| **Embedding model** | ResNet18 pretrained (ImageNet), FC removed, 512-dim, L2-normalized |
| **Build pipeline** | `build_embeddings.py` → one pass over 44,419 images (~10 min on CPU) |
| **Graph build** | `build_graph.py` → batched einsum, ~25 sec for 44K nodes |
| **FAISS build** | `build_faiss.py` → k-means + IVF, under 10 seconds |
| **Serving** | FastAPI + Uvicorn, `KMP_DUPLICATE_LIB_OK=TRUE` for Mac OpenMP fix |
| **UI** | Streamlit 12-page interface, Matplotlib charts, NetworkX graph viz |
| **RAG** | Anthropic Claude claude-sonnet-4, context = retrieved product metadata |
| **Evaluation** | `benchmark.py` — Recall@k + latency median/p95/p99 across all 3 methods |
| **Hardware** | Apple M-series CPU, no GPU required for inference |
| **Dataset** | Kaggle Fashion Product Images — 44,419 products, styles.csv metadata |
| **Artifacts** | embeddings.npy (86 MB), knn_graph.pkl (87 MB), faiss.index (87 MB) |

---

## 🚨 Postmortem

```
╔══════════════════════════╦═══════════════════════════════╦══════════════════════════════════╦════════════════════════════════╗
║  Issue                   ║  Root Cause                   ║  Fix                             ║  Lesson                        ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  FAISS crashes on Mac    ║  PyTorch + FAISS both link    ║  KMP_DUPLICATE_LIB_OK=TRUE       ║  Check OpenMP conflicts on     ║
║                          ║  libomp.dylib (OpenMP clash)  ║  set before uvicorn startup      ║  Apple Silicon                 ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  Benchmark times out     ║  3 sequential API calls for   ║  Increase Streamlit request      ║  Set generous timeouts for     ║
║                          ║  compare-all exceed 30s       ║  timeout to 120s                 ║  multi-endpoint benchmarks     ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  Wrong API served        ║  Two main.py files in sibling ║  Explicit port 8001 per project  ║  Always use explicit ports;    ║
║                          ║  folders; uvicorn picks wrong ║  uvicorn main:app --port 8001    ║  never rely on defaults        ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  Graph slower than       ║  44K products with hash       ║  Documented honestly — graph     ║  Report real numbers; the      ║
║  baseline at 44K scale   ║  pre-filter too small for     ║  wins at 1M+ scale               ║  graph advantage emerges       ║
║                          ║  graph BFS overhead to pay    ║                                  ║  at larger dataset sizes       ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  HuggingFace image 403   ║  HF servers block GitHub CDN  ║  Images served via local         ║  Free hosting has constraints; ║
║                          ║  raw.githubusercontent.com    ║  FastAPI /image/{id} endpoint    ║  design for local-first        ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  HuggingFace LFS rate    ║  44K images = 44K LFS objects ║  Reorganize into subdirs of      ║  Always check hosting limits   ║
║  limit (1000 req/5 min)  ║  exceeds free-tier limit      ║  10K files each before upload    ║  before designing data layout  ║
╠══════════════════════════╬═══════════════════════════════╬══════════════════════════════════╬════════════════════════════════╣
║  GitHub large file warn  ║  embeddings.npy (86 MB) and   ║  Added to .gitignore; document   ║  Always gitignore large binary ║
║                          ║  faiss.index (87 MB) > 50 MB  ║  rebuild steps in README         ║  artifacts (models, indexes)   ║
╚══════════════════════════╩═══════════════════════════════╩══════════════════════════════════╩════════════════════════════════╝
```

---

## 👥 Team Contributions

```
╔════════════════════════════════════════════╦══════════════════════════════════════════════╗
║  Akilan Manivannan (100863473)             ║  Akila Lourdes Miriyala Francis (100863383)  ║
║  GitHub: AkilanManivannanak               ║  GitHub: AKilalours                          ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  DATA PIPELINE                            ║  DATA PIPELINE                               ║
║  • Kaggle dataset acquisition             ║  • styles.csv metadata parsing & cleaning    ║
║  • Image validation and filtering         ║  • Product ID to filepath index mapping      ║
║  • Dataset split strategy                 ║  • Embedding pipeline orchestration          ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  DATA STRUCTURES                          ║  DATA STRUCTURES                             ║
║  • k-NN Graph design and build            ║  • Hash Table (hash_index.py)                ║
║  • knn_graph.py adjacency list            ║  • Nested Hash Table (color_index.py)        ║
║  • BFS traversal implementation           ║  • Brand Hash Index (brand_index.py)         ║
║  • Graph save/load (pickle)               ║  • Min-Heap ranker (heap_ranker.py)          ║
║  • MMR reranker (mmr_reranker.py)        ║  • Searcher unified interface (searcher.py)  ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  ML / RETRIEVAL                           ║  ML / RETRIEVAL                              ║
║  • ResNet18 embedder (embedder.py)        ║  • FAISS IVF index (faiss_index.py)          ║
║  • build_embeddings.py pipeline           ║  • build_faiss.py pipeline                   ║
║  • build_graph.py pipeline                ║  • Baseline brute-force cosine search        ║
║  • Graph search method                    ║  • L2 normalization strategy                 ║
║  • Style Transfer search module           ║  • CLIP fusion search module                 ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  BACKEND / API                            ║  BACKEND / API                               ║
║  • /search/by_id endpoint                 ║  • /search/upload endpoint                   ║
║  • /graph_neighbors BFS endpoint          ║  • /benchmark endpoint                       ║
║  • /image file serving                    ║  • /categories + /colors + /brands           ║
║  • /brand_compare endpoint                ║  • CORS + middleware setup                   ║
║  • /outfit/complete endpoint              ║  • /trending + /search_history               ║
║  • FastAPI routing + error handling       ║  • /realtime/add live ingestion              ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  UI / FRONTEND                            ║  UI / FRONTEND                               ║
║  • Graph Visualization page               ║  • Search page (upload + ID + URL + multi)  ║
║  • NetworkX + Matplotlib graph            ║  • Benchmark Charts page                     ║
║  • Brand Compare page                     ║  • Live latency + recall bar charts          ║
║  • Visual DNA / GradCAM page              ║  • Product cards with score bars             ║
║  • Style Transfer page                    ║  • Outfit Completion page                    ║
║  • Sidebar system stats panel             ║  • AI Assistant page (RAG + Claude)          ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  EVALUATION                               ║  EVALUATION                                  ║
║  • benchmark.py architecture              ║  • Recall@k metric implementation            ║
║  • Latency measurement (p50/p95/p99)      ║  • Results CSV export                        ║
║  • Multi-method comparison logic          ║  • Interpretation + documentation            ║
╠════════════════════════════════════════════╬══════════════════════════════════════════════╣
║  MLOPS / DEPLOYMENT                       ║  MLOPS / DEPLOYMENT                          ║
║  • GitHub repo setup                      ║  • GitHub repo (AKilalours)                  ║
║  • HuggingFace Spaces deployment          ║  • SSH key configuration                     ║
║  • Docker containerization               ║  • KMP_DUPLICATE_LIB_OK fix discovery        ║
║  • Port isolation (8001 / 8502)           ║  • README + submission documentation         ║
║  • ngrok public tunneling                 ║  • Postmortem authoring                      ║
╚════════════════════════════════════════════╩══════════════════════════════════════════════╝
```

---

<div align="center">

```
⚡ 1.79 ms FAISS  ·  8.63 ms Baseline  ·  Recall@10 = 0.900
44,419 Products  ·  467 Brands  ·  47 Colors  ·  444,190 Graph Edges
512-dim ResNet18  ·  CLIP Fusion  ·  RAG Assistant  ·  GradCAM Explainability
Hash Table + Nested Color Index + Brand Index + k-NN Graph + Min-Heap + FAISS IVF + MMR
FastAPI · Streamlit · NetworkX · Matplotlib · PyTorch · Anthropic Claude · Apple Silicon
```

**CS 631 — Algorithms & Data Structures · Prof. Dr. Abla Bedoui · LIU Brooklyn · Spring 2026**

**Akila Lourdes Miriyala Francis (100863383) · Akilan Manivannan (100863473)**

</div>
