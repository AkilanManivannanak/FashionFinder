"""
ui.py  -  FashionFinder Advanced UI
-------------------------------------
Features:
  - 3-method retrieval: Baseline | k-NN Graph | FAISS ANN
  - Side-by-side comparison of all 3 methods
  - Live benchmark bar charts (latency + recall)
  - Color filter (nested hash index)
  - Graph visualization (product similarity network)
  - Clean product cards with similarity score progress bars
"""

import streamlit as st
import requests
import os
import io
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

API_BASE = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="FashionFinder",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.product-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 10px;
    margin: 4px;
    border: 1px solid #333;
    text-align: center;
}
.score-bar-wrap {
    background: #333;
    border-radius: 6px;
    height: 6px;
    margin-top: 4px;
}
.metric-box {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #444;
}
.method-header {
    font-size: 15px;
    font-weight: bold;
    padding: 6px 12px;
    border-radius: 8px;
    margin-bottom: 8px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#4FC3F7; font-size:42px; margin-bottom:0;'>
FashionFinder
</h1>
<p style='text-align:center; color:#aaa; font-size:15px; margin-top:4px;'>
Visual Search and Image Retrieval System<br>
<span style='color:#81C784;'>ResNet18 Embeddings</span> &nbsp;|&nbsp;
<span style='color:#FFB74D;'>k-NN Graph</span> &nbsp;|&nbsp;
<span style='color:#F48FB1;'>FAISS ANN</span> &nbsp;|&nbsp;
<span style='color:#CE93D8;'>Min-Heap Ranking</span> &nbsp;|&nbsp;
<span style='color:#80DEEA;'>Hash Table Index</span>
</p>
<hr style='border-color:#333;'>
""", unsafe_allow_html=True)

# ── Fetch API metadata ────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=60)
def fetch_categories():
    try:
        r = requests.get(f"{API_BASE}/categories", timeout=3)
        return r.json().get("categories", [])
    except:
        return []

@st.cache_data(ttl=60)
def fetch_colors(category=None):
    try:
        params = {"category": category} if category else {}
        r = requests.get(f"{API_BASE}/colors", params=params, timeout=3)
        return r.json().get("colors", [])
    except:
        return []

health = fetch_health()
categories = fetch_categories()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Search Settings")

    query_type = st.radio("Query type", ["Upload Image", "Search by Product ID"])

    k = st.slider("Top-k results", 1, 20, 10)

    method = st.radio(
        "Retrieval method",
        ["graph", "baseline", "faiss", "compare all 3"],
        help="Graph=k-NN traversal | Baseline=brute-force | FAISS=ANN index | Compare=all 3 side by side"
    )

    st.markdown("---")
    st.markdown("### Filters")

    cat_options = ["All"] + [c["name"] for c in categories]
    selected_cat = st.selectbox("Category (Hash Table)", cat_options)
    category_param = None if selected_cat == "All" else selected_cat

    colors = fetch_colors(category_param)
    color_options = ["All"] + colors
    selected_color = st.selectbox("Color (Nested Hash Index)", color_options)
    color_param = None if selected_color == "All" else selected_color

    st.markdown("---")

    # Stats panel
    if health:
        st.markdown("### System Stats")
        st.markdown(f"**Products indexed:** {health.get('products_indexed', 0):,}")
        st.markdown(f"**Graph loaded:** {'Yes' if health.get('graph_loaded') else 'No'}")
        st.markdown(f"**FAISS loaded:** {'Yes' if health.get('faiss_loaded') else 'No'}")
        cats = health.get('categories', [])
        st.markdown(f"**Categories:** {len(cats)}")

    st.markdown("---")
    with st.expander("Data Structures", expanded=False):
        st.markdown("""
**Hash Table** `hash_index.py`
Maps category → indices. O(1) lookup. Reduces search space ~70%.

**Nested Hash Table** `color_index.py`
Maps category → color → indices. Two-level filtering.

**k-NN Graph** `knn_graph.py`
Adjacency list. Each node connects to 10 nearest neighbors. Graph traversal at query time.

**Min-Heap** `heap_ranker.py`
Priority queue. Tracks top-k in O(n log k) vs O(n log n) for full sort.

**FAISS IVF** `faiss_index.py`
Inverted file index. Clusters embeddings into 100 Voronoi cells. Sub-linear ANN search.
        """)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_benchmark, tab_graph = st.tabs([
    "Search", "Benchmark Charts", "Graph Visualization"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:

    uploaded = None
    product_id = None

    if query_type == "Upload Image":
        uploaded = st.file_uploader(
            "Upload a fashion product image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            col_img, col_info = st.columns([1, 3])
            with col_img:
                img = Image.open(uploaded)
                st.image(img, caption="Query Image", use_container_width=True)
            with col_info:
                st.markdown(f"**File:** {uploaded.name}")
                st.markdown(f"**Size:** {uploaded.size / 1024:.1f} KB")
                st.markdown(f"**Method:** {method}")
                if category_param:
                    st.markdown(f"**Category filter:** {category_param}")
                if color_param:
                    st.markdown(f"**Color filter:** {color_param}")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            product_id = st.number_input("Product ID", min_value=1, value=1163, step=1)
        with col2:
            if st.button("Preview", use_container_width=True):
                try:
                    r = requests.get(f"{API_BASE}/product/{int(product_id)}", timeout=5)
                    if r.status_code == 200:
                        meta = r.json()
                        img_r = requests.get(f"{API_BASE}/image/{int(product_id)}", timeout=5)
                        if img_r.status_code == 200:
                            img = Image.open(io.BytesIO(img_r.content))
                            st.image(img, width=120)
                        st.caption(f"{meta.get('name','')} | {meta.get('articleType','')} | {meta.get('baseColour','')}")
                except Exception as e:
                    st.error(str(e))

    search_btn = st.button("Search", type="primary", use_container_width=True)

    # ── Result rendering helper ────────────────────────────────────────────────
    def render_results(results, latency_ms, method_label, color):
        st.markdown(
            f"<div style='background:{color}22; border:1px solid {color}; border-radius:8px; "
            f"padding:8px 14px; margin-bottom:10px;'>"
            f"<span style='color:{color}; font-weight:bold;'>{method_label}</span>"
            f"&nbsp;&nbsp; Latency: <code>{latency_ms} ms</code>"
            f"&nbsp;&nbsp; Results: <code>{len(results)}</code>"
            f"</div>",
            unsafe_allow_html=True
        )

        cols = st.columns(5)
        for i, item in enumerate(results[:10]):
            with cols[i % 5]:
                pid = item.get("id")
                try:
                    img_r = requests.get(f"{API_BASE}/image/{pid}", timeout=3)
                    if img_r.status_code == 200:
                        img = Image.open(io.BytesIO(img_r.content))
                        st.image(img, use_container_width=True)
                    else:
                        st.markdown("🖼️")
                except:
                    st.markdown("🖼️")

                score = item.get("score", 0)
                name  = item.get("name", "")[:22]
                atype = item.get("articleType", "")
                bclr  = item.get("baseColour", "")

                # Score progress bar
                bar_pct = int(score * 100)
                st.markdown(
                    f"<div style='font-size:11px; color:#eee; margin-top:4px;'>"
                    f"<b>#{item['rank']}</b> {name}</div>"
                    f"<div style='font-size:10px; color:#aaa;'>{atype} | {bclr}</div>"
                    f"<div style='background:#333; border-radius:4px; height:5px; margin-top:4px;'>"
                    f"<div style='background:{color}; width:{bar_pct}%; height:5px; border-radius:4px;'></div>"
                    f"</div>"
                    f"<div style='font-size:10px; color:{color};'>Score: {score:.3f}</div>",
                    unsafe_allow_html=True
                )

    # ── Search execution ───────────────────────────────────────────────────────
    if search_btn:
        if query_type == "Upload Image" and uploaded is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Searching..."):
                try:
                    params_base = {"k": k, "category": category_param, "color": color_param}

                    if method == "compare all 3":
                        # Run all 3 methods
                        all_results = {}
                        method_configs = [
                            ("baseline", "#4FC3F7", "Baseline (Brute-Force)"),
                            ("graph",    "#81C784", "Graph (k-NN Traversal)"),
                            ("faiss",    "#F48FB1", "FAISS (ANN Index)"),
                        ]

                        for m_key, m_color, m_label in method_configs:
                            p = {**params_base, "method": m_key}
                            if query_type == "Upload Image":
                                uploaded.seek(0)
                                files = {"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")}
                                r = requests.post(f"{API_BASE}/search/upload", files=files, params=p, timeout=120)
                            else:
                                r = requests.post(f"{API_BASE}/search/by_id",
                                                  params={**p, "product_id": int(product_id)}, timeout=120)
                            all_results[m_key] = r.json()

                        # Display 3 columns
                        st.markdown("### Results - All 3 Methods")
                        c1, c2, c3 = st.columns(3)
                        cols_map = {"baseline": c1, "graph": c2, "faiss": c3}

                        for m_key, m_color, m_label in method_configs:
                            data = all_results[m_key]
                            with cols_map[m_key]:
                                render_results(data["results"], data["latency_ms"], m_label, m_color)

                        # Speed comparison
                        st.markdown("---")
                        st.markdown("### Latency Comparison")
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        b_lat = all_results["baseline"]["latency_ms"]
                        g_lat = all_results["graph"]["latency_ms"]
                        f_lat = all_results["faiss"]["latency_ms"]
                        fastest = min(b_lat, g_lat, f_lat)

                        with mc1:
                            st.metric("Baseline", f"{b_lat} ms")
                        with mc2:
                            st.metric("Graph", f"{g_lat} ms",
                                      delta=f"{b_lat - g_lat:+.1f} ms vs baseline")
                        with mc3:
                            st.metric("FAISS", f"{f_lat} ms",
                                      delta=f"{b_lat - f_lat:+.1f} ms vs baseline")
                        with mc4:
                            fastest_name = ["Baseline","Graph","FAISS"][[b_lat,g_lat,f_lat].index(fastest)]
                            st.metric("Fastest", fastest_name, f"{fastest} ms")

                        # Store for benchmark tab
                        st.session_state["last_comparison"] = {
                            "baseline": all_results["baseline"],
                            "graph":    all_results["graph"],
                            "faiss":    all_results["faiss"],
                        }

                    else:
                        # Single method
                        p = {**params_base, "method": method}
                        if query_type == "Upload Image":
                            uploaded.seek(0)
                            files = {"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")}
                            r = requests.post(f"{API_BASE}/search/upload", files=files, params=p, timeout=120)
                        else:
                            r = requests.post(f"{API_BASE}/search/by_id",
                                              params={**p, "product_id": int(product_id)}, timeout=120)

                        data = r.json()
                        colors_map = {"baseline": "#4FC3F7", "graph": "#81C784", "faiss": "#F48FB1"}
                        labels_map = {
                            "baseline": "Baseline (Brute-Force)",
                            "graph":    "Graph (k-NN Traversal)",
                            "faiss":    "FAISS (ANN Index)"
                        }
                        st.markdown(f"### Results - {labels_map.get(method, method)}")
                        render_results(data["results"], data["latency_ms"],
                                       labels_map.get(method, method),
                                       colors_map.get(method, "#fff"))

                        st.session_state[f"last_{method}"] = data

                except Exception as e:
                    st.error(f"Search failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BENCHMARK CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("### Live Benchmark - Run All 3 Methods")
    st.markdown("Click the button to run a fresh benchmark across N random products from the dataset.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        bench_n = st.number_input("Number of queries", min_value=10, max_value=500, value=50, step=10)
    with col_b:
        run_bench = st.button("Run Benchmark", type="primary")

    if run_bench:
        with st.spinner(f"Running {bench_n} queries across all 3 methods..."):
            try:
                r = requests.post(
                    f"{API_BASE}/benchmark",
                    params={"n": bench_n, "k": k},
                    timeout=1200
                )
                bench_data = r.json()
                st.session_state["bench_data"] = bench_data
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

    if "bench_data" in st.session_state:
        bd = st.session_state["bench_data"]

        # ── Metric cards ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Latency Results")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metrics = [
            (c1, "Baseline Median", f"{bd['baseline_median_ms']} ms", "#4FC3F7"),
            (c2, "Graph Median",    f"{bd['graph_median_ms']} ms",    "#81C784"),
            (c3, "FAISS Median",    f"{bd['faiss_median_ms']} ms",    "#F48FB1"),
            (c4, "Baseline p95",    f"{bd['baseline_p95_ms']} ms",    "#4FC3F7"),
            (c5, "Graph p95",       f"{bd['graph_p95_ms']} ms",       "#81C784"),
            (c6, "FAISS p95",       f"{bd['faiss_p95_ms']} ms",       "#F48FB1"),
        ]
        for col, label, val, color in metrics:
            with col:
                st.markdown(
                    f"<div style='background:#1e1e2e; border:1px solid {color}; border-radius:10px; "
                    f"padding:12px; text-align:center;'>"
                    f"<div style='color:#aaa; font-size:11px;'>{label}</div>"
                    f"<div style='color:{color}; font-size:22px; font-weight:bold;'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ── Charts side by side ───────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Latency Comparison (ms)")
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
            ax.set_facecolor("#1e1e2e")

            methods  = ["Baseline", "Graph", "FAISS"]
            medians  = [bd["baseline_median_ms"], bd["graph_median_ms"], bd["faiss_median_ms"]]
            p95s     = [bd["baseline_p95_ms"],    bd["graph_p95_ms"],    bd["faiss_p95_ms"]]
            colors_b = ["#4FC3F7", "#81C784", "#F48FB1"]

            x = np.arange(len(methods))
            w = 0.35
            bars1 = ax.bar(x - w/2, medians, w, label="Median", color=colors_b, alpha=0.9)
            bars2 = ax.bar(x + w/2, p95s,    w, label="p95",    color=colors_b, alpha=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, color="white")
            ax.set_ylabel("Latency (ms)", color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1e1e2e", labelcolor="white")
            ax.spines[:].set_color("#444")

            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f"{bar.get_height():.1f}", ha="center", va="bottom",
                        color="white", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with chart_col2:
            st.markdown("#### Recall@k Comparison")
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
            ax2.set_facecolor("#1e1e2e")

            recalls = [1.000, bd["graph_recall"], bd["faiss_recall"]]
            bars = ax2.bar(methods, recalls, color=colors_b, alpha=0.9, width=0.5)

            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel("Recall@k", color="white")
            ax2.tick_params(colors="white")
            ax2.spines[:].set_color("#444")
            ax2.set_xticklabels(methods, color="white")
            ax2.axhline(y=1.0, color="#666", linestyle="--", linewidth=1)

            for bar, val in zip(bars, recalls):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom",
                         color="white", fontsize=11, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Full Results Table")
        df_bench = pd.DataFrame({
            "Method":        ["Baseline (Brute-Force)", "Graph (k-NN)", "FAISS (ANN)"],
            "Median (ms)":   [bd["baseline_median_ms"], bd["graph_median_ms"], bd["faiss_median_ms"]],
            "p95 (ms)":      [bd["baseline_p95_ms"],    bd["graph_p95_ms"],    bd["faiss_p95_ms"]],
            "p99 (ms)":      [bd["baseline_p99_ms"],    bd["graph_p99_ms"],    bd["faiss_p99_ms"]],
            "Recall@k":      [1.000,                    bd["graph_recall"],    bd["faiss_recall"]],
            "Search Type":   ["Exact", "Approximate", "Approximate"],
        })
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.markdown(f"""
**Interpretation:**
- Baseline is exact but scans all candidates every time
- Graph traverses prebuilt neighbor links, trading small recall loss for structure
- FAISS uses cluster-based ANN, fastest at scale with minimal recall loss
- Hash Table + Color Index pre-filter reduces candidates before any method runs
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: GRAPH VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.markdown("### Product Similarity Graph")
    st.markdown("Shows how products connect to their nearest neighbors in the k-NN graph.")

    col_gv1, col_gv2 = st.columns([1, 2])
    with col_gv1:
        center_id = st.number_input("Center Product ID", min_value=1, value=1163, step=1)
        hops      = st.slider("Graph hops to explore", 1, 3, 2)
        show_btn  = st.button("Show Graph", type="primary")

    if show_btn:
        with st.spinner("Fetching graph neighbors..."):
            try:
                r = requests.get(
                    f"{API_BASE}/graph_neighbors/{int(center_id)}",
                    params={"hops": hops},
                    timeout=15
                )
                gdata = r.json()

                G = nx.Graph()
                node_labels = {}
                node_colors = []
                node_sizes  = []

                center_node = gdata["center"]["id"]
                G.add_node(center_node)
                node_labels[center_node] = f"#{center_node}\n{gdata['center'].get('articleType','')[:10]}"

                # Color by category
                cat_color_map = {
                    "Apparel":       "#81C784",
                    "Footwear":      "#FFB74D",
                    "Accessories":   "#F48FB1",
                    "Personal Care": "#CE93D8",
                    "Sporting Goods":"#80DEEA",
                    "Free Items":    "#FFF176",
                    "Home":          "#FFCC80",
                }

                def get_node_color(cat):
                    return cat_color_map.get(cat, "#90A4AE")

                center_cat = gdata["center"].get("masterCategory", "Unknown")

                for node_data in gdata.get("nodes", []):
                    nid  = node_data["id"]
                    cat  = node_data.get("masterCategory", "Unknown")
                    atype = node_data.get("articleType", "")[:10]
                    score = node_data.get("score", 0)
                    depth = node_data.get("depth", 1)

                    G.add_node(nid)
                    G.add_edge(node_data["parent"], nid, weight=score)
                    node_labels[nid] = f"{atype}\n{score:.2f}"

                # Draw
                fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0e1117")
                ax.set_facecolor("#0e1117")

                all_nodes = list(G.nodes())
                n_colors  = []
                n_sizes   = []

                for n in all_nodes:
                    if n == center_node:
                        n_colors.append("#FFD700")
                        n_sizes.append(800)
                    else:
                        # find cat from nodes list
                        cat = "Unknown"
                        for nd in gdata.get("nodes", []):
                            if nd["id"] == n:
                                cat = nd.get("masterCategory", "Unknown")
                                break
                        n_colors.append(get_node_color(cat))
                        n_sizes.append(300)

                pos = nx.spring_layout(G, seed=42, k=1.5)
                nx.draw_networkx_nodes(G, pos, node_color=n_colors, node_size=n_sizes, ax=ax, alpha=0.9)
                nx.draw_networkx_edges(G, pos, edge_color="#444", ax=ax, alpha=0.6, width=1.2)
                nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7,
                                        font_color="white", ax=ax)

                # Legend
                legend_patches = [mpatches.Patch(color=c, label=cat)
                                  for cat, c in cat_color_map.items()]
                legend_patches.append(mpatches.Patch(color="#FFD700", label="Query (center)"))
                ax.legend(handles=legend_patches, loc="upper left",
                          facecolor="#1e1e2e", labelcolor="white", fontsize=8)

                ax.axis("off")
                ax.set_title(
                    f"k-NN Similarity Graph: Product {center_id} | {hops} hops | {G.number_of_nodes()} nodes",
                    color="white", fontsize=12, pad=10
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Node table
                st.markdown("---")
                st.markdown("#### Connected Products")
                rows = []
                for nd in gdata.get("nodes", []):
                    rows.append({
                        "Product ID":   nd["id"],
                        "Name":         nd.get("name", "")[:40],
                        "Category":     nd.get("masterCategory", ""),
                        "Article Type": nd.get("articleType", ""),
                        "Color":        nd.get("baseColour", ""),
                        "Similarity":   round(nd.get("score", 0), 4),
                        "Depth":        nd.get("depth", 1),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Graph visualization failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#555; font-size:12px;'>
FashionFinder | Visual Search System | ResNet18 + k-NN Graph + FAISS ANN + Min-Heap + Hash Table<br>
Built by Akila Lourdes Miriyala Francis | Data Structures Course Project
</p>
""", unsafe_allow_html=True)
