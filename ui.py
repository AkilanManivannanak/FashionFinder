"""
ui.py - FashionFinder v4.0 — Professional Edition
"""
import streamlit as st
import requests, io, os
import numpy as np, pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

API = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="FashionFinder",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #080c14; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0c1220 !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #c9d4e8 !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #1e4976) !important;
    color: #7ec8e3 !important;
    border: 1px solid #2a5a8c !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    transition: all 0.2s ease !important;
    padding: 6px 12px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e4976, #2361a0) !important;
    border-color: #4FC3F7 !important;
    color: #4FC3F7 !important;
}
[data-testid="baseButton-primary"] > button,
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0066cc, #0052a3) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(0,102,204,0.3) !important;
}
[data-testid="baseButton-primary"] > button:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0077ee, #0066cc) !important;
    box-shadow: 0 4px 20px rgba(0,102,204,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #0066cc !important;
    box-shadow: 0 0 0 2px rgba(0,102,204,0.2) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #0066cc !important;
}

/* Toggle */
.stToggle > label { color: #94a3b8 !important; }

/* Dataframe */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0c1220; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a5a8c; }

/* Metric */
[data-testid="metric-container"] {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Colors ─────────────────────────────────────────────────────────────────────
C = {
    "blue":   "#4FC3F7",
    "green":  "#4ade80",
    "pink":   "#f472b6",
    "purple": "#a78bfa",
    "gold":   "#fbbf24",
    "orange": "#fb923c",
    "teal":   "#2dd4bf",
    "red":    "#f87171",
    "gray":   "#64748b",
    "text":   "#e2e8f0",
    "dim":    "#94a3b8",
    "bg":     "#080c14",
    "card":   "#0f1a2e",
    "border": "#1e3a5f",
}

# ── API helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def api(endpoint, **params):
    try: return requests.get(f"{API}/{endpoint}", params=params, timeout=5).json()
    except: return {}

health     = api("health")
categories = [c["name"] for c in api("categories").get("categories", [])]
all_colors = api("colors").get("colors", [])
all_brands = [b["name"] for b in api("brands", top_n=150).get("brands", [])]

# ── Session state ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Search"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_products" not in st.session_state:
    st.session_state.chat_products = []
if "clip_style" not in st.session_state:
    st.session_state.clip_style = ""

PAGES = ["Search","Brands","Style","Outfit","DNA","CLIP","Add","Benchmark","Graph","Timeline","Trending","Assistant"]
PAGE_ICONS = {"Search":"🔍","Brands":"🏷️","Style":"🎨","Outfit":"👔","DNA":"🧬",
              "CLIP":"🤖","Add":"➕","Benchmark":"📊","Graph":"🕸️",
              "Timeline":"📅","Trending":"🔥","Assistant":"💬"}

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:24px 0 0;text-align:center;">
  <div style="display:inline-flex;align-items:center;gap:12px;margin-bottom:4px;">
    <span style="font-size:32px;">👗</span>
    <span style="font-size:36px;font-weight:900;letter-spacing:-1px;
      background:linear-gradient(135deg,{C['blue']},{C['teal']},{C['green']});
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      FashionFinder
    </span>
    <span style="font-size:12px;color:{C['gold']};font-weight:700;
      background:#1a150a;border:1px solid #3a2a0a;border-radius:6px;
      padding:3px 8px;letter-spacing:1px;">v4.0</span>
  </div>
  <p style="color:{C['dim']};font-size:12px;margin:0;letter-spacing:2px;text-transform:uppercase;">
    Visual Search · 44,419 Products · 467 Brands · ResNet18 · FAISS · k-NN Graph
  </p>
</div>
""", unsafe_allow_html=True)

# ── NAV BAR ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="height:1px;background:linear-gradient(90deg,transparent,{C['border']},transparent);
margin:16px 0 12px;"></div>
""", unsafe_allow_html=True)

nav_cols = st.columns(len(PAGES))
for i, (col, page) in enumerate(zip(nav_cols, PAGES)):
    with col:
        icon = PAGE_ICONS.get(page, "")
        active = st.session_state.page == page
        if st.button(
            f"{icon} {page}" if not active else f"{icon} {page}",
            key=f"nav_{i}",
            use_container_width=True,
            type="primary" if active else "secondary"
        ):
            st.session_state.page = page
            st.rerun()

st.markdown(f"""
<div style="height:1px;background:linear-gradient(90deg,transparent,{C['border']},transparent);
margin:12px 0 20px;"></div>
""", unsafe_allow_html=True)

ACTIVE = st.session_state.page

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo in sidebar
    st.markdown(f"""
    <div style="padding:16px 0 20px;border-bottom:1px solid {C['border']};margin-bottom:20px;">
      <div style="font-size:20px;font-weight:800;color:{C['blue']};">⚙️ Controls</div>
      <div style="font-size:11px;color:{C['dim']};margin-top:2px;">Adjust search settings</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px;font-weight:700;color:{C['dim']};letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Retrieval Method</div>", unsafe_allow_html=True)
    method = st.radio("", ["faiss","graph","baseline"],
                      format_func=lambda x: {
                          "faiss":    "⚡  FAISS  —  Fastest (1.79ms)",
                          "graph":   "🕸️  Graph  —  Scalable",
                          "baseline":"🎯  Baseline  —  Exact"
                      }[x], label_visibility="collapsed")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    k = st.slider("Top-k results", 1, 20, 10)
    use_mmr = st.toggle("MMR Diversity", help="Reduces duplicate results")
    if use_mmr:
        mmr_lambda = st.slider("Diversity ←→ Relevance", 0.0, 1.0, 0.6, 0.05)
    else:
        mmr_lambda = 0.6

    st.markdown(f"""
    <div style="height:1px;background:{C['border']};margin:16px 0;"></div>
    <div style='font-size:11px;font-weight:700;color:{C['dim']};letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Filters</div>
    """, unsafe_allow_html=True)

    sel_cat    = st.selectbox("Category", ["All"] + categories, label_visibility="collapsed")
    cat_param  = None if sel_cat == "All" else sel_cat
    fcolors    = api("colors", category=cat_param).get("colors", all_colors) if cat_param else all_colors
    sel_color  = st.selectbox("Color", ["All"] + fcolors, label_visibility="collapsed")
    color_param = None if sel_color == "All" else sel_color
    sel_brand  = st.selectbox("Brand", ["All"] + all_brands, label_visibility="collapsed")
    brand_param = None if sel_brand == "All" else sel_brand

    st.markdown(f"""
    <div style="height:1px;background:{C['border']};margin:16px 0;"></div>
    <div style='font-size:11px;font-weight:700;color:{C['dim']};letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;'>System</div>
    """, unsafe_allow_html=True)

    g  = health.get("graph_loaded", False)
    f  = health.get("faiss_loaded", False)
    cl = health.get("clip_loaded",  False)
    rt = health.get("realtime_buffer_size", 0)

    sc = st.columns(3)
    for col, label, val, color in [
        (sc[0], "Products", f"{health.get('products_indexed',0):,}", C['blue']),
        (sc[1], "Brands",   f"{health.get('brands_detected',0):,}",  C['gold']),
        (sc[2], "Searches", f"{health.get('total_searches',0):,}",   C['green']),
    ]:
        with col:
            st.markdown(
                f"<div style='background:{C['card']};border:1px solid {C['border']};"
                f"border-radius:8px;padding:8px 6px;text-align:center;'>"
                f"<div style='color:{C['dim']};font-size:9px;font-weight:600;text-transform:uppercase;'>{label}</div>"
                f"<div style='color:{color};font-size:15px;font-weight:800;'>{val}</div></div>",
                unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin-top:10px;font-size:11px;color:{C['dim']};'>"
        f"Graph {'<span style=\"color:#4ade80\">●</span>' if g else '<span style=\"color:#f87171\">●</span>'} &nbsp;"
        f"FAISS {'<span style=\"color:#4ade80\">●</span>' if f else '<span style=\"color:#f87171\">●</span>'} &nbsp;"
        f"CLIP {'<span style=\"color:#4ade80\">●</span>' if cl else '<span style=\"color:#fbbf24\">●</span>'}"
        f"</div>", unsafe_allow_html=True)

    if rt > 0:
        st.markdown(f"<div style='color:{C['red']};font-size:11px;margin-top:6px;'>🔴 {rt} live products</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="height:1px;background:{C['border']};margin:16px 0;"></div>
    """, unsafe_allow_html=True)
    with st.expander("🧩 Data Structures"):
        for ds, role, color in [
            ("Hash Table",        "O(1) category → indices",   C['blue']),
            ("Nested Hash Table", "O(1) color+category filter", C['teal']),
            ("Brand Index",       "467 brands, 3-level hash",  C['gold']),
            ("k-NN Graph",        "444K edges, BFS O(log n)",  C['pink']),
            ("Min-Heap",          "Top-k in O(n log k)",       C['purple']),
            ("FAISS IVF",         "100 clusters, sub-linear",  C['green']),
        ]:
            st.markdown(
                f"<div style='margin-bottom:8px;'>"
                f"<span style='color:{color};font-size:11px;font-weight:700;'>{ds}</span><br>"
                f"<span style='color:{C['dim']};font-size:10px;'>{role}</span></div>",
                unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def card(item, col, show_exp=True):
    with col:
        pid   = item.get("id"); score = item.get("score", 0)
        name  = item.get("name","")[:28]; atype = item.get("articleType","")
        clr   = item.get("baseColour",""); brand = item.get("brand","")
        expl  = item.get("explanation",""); rank  = item.get("rank","")
        is_rt = item.get("is_realtime", False)

        try:
            r = requests.get(f"{API}/image/{pid}", timeout=3)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                st.image(img, use_container_width=True)
        except: st.markdown("🖼️")

        bar = int(score * 100)
        st.markdown(
            f"<div style='margin-top:6px;'>"
            f"<div style='font-size:11px;font-weight:700;color:{C['text']};line-height:1.3;'>"
            f"{'🔴 ' if is_rt else ''}#{rank} {name}</div>"
            f"<div style='font-size:10px;color:{C['dim']};margin-top:1px;'>{atype} · {clr}</div>"
            f"<div style='margin:4px 0 2px;'>"
            f"<span style='background:linear-gradient(90deg,#1a150a,#2a2000);color:{C['gold']};"
            f"border:1px solid #3a2a0a;border-radius:6px;padding:1px 7px;font-size:10px;font-weight:700;'>🏷 {brand}</span>"
            f"</div>"
            f"<div style='background:#1a2540;border-radius:3px;height:3px;margin:4px 0 2px;'>"
            f"<div style='background:linear-gradient(90deg,{C['blue']},{C['teal']});width:{bar}%;height:3px;border-radius:3px;'></div></div>"
            f"<div style='font-size:10px;color:{C['teal']};font-weight:600;'>{score:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True)

        if show_exp and expl:
            st.markdown(
                f"<div style='background:#091828;border-left:2px solid {C['blue']};border-radius:0 4px 4px 0;"
                f"padding:4px 8px;margin-top:4px;font-size:10px;color:{C['teal']};'>💡 {expl}</div>",
                unsafe_allow_html=True)

def status(label, latency, n, color, mmr=False):
    mmr_tag = f"&nbsp;·&nbsp;<span style='color:{C['green']};'>MMR ON</span>" if mmr else ""
    st.markdown(
        f"<div style='background:linear-gradient(90deg,#091828,#091820);border:1px solid {color}33;"
        f"border-left:3px solid {color};border-radius:8px;padding:10px 16px;margin-bottom:16px;'>"
        f"<span style='color:{color};font-weight:700;font-size:13px;'>{label}</span>"
        f"&nbsp;&nbsp;<code style='background:#0a1628;color:{C['blue']};padding:2px 6px;border-radius:4px;'>{latency} ms</code>"
        f"&nbsp;&nbsp;<code style='background:#0a1628;color:{C['text']};padding:2px 6px;border-radius:4px;'>{n} results</code>"
        f"{mmr_tag}</div>",
        unsafe_allow_html=True)

def section(title, subtitle=""):
    st.markdown(
        f"<div style='margin-bottom:20px;'>"
        f"<h2 style='font-size:22px;font-weight:800;color:{C['text']};margin:0;letter-spacing:-0.5px;'>{title}</h2>"
        f"{'<p style=\"color:'+C['dim']+';font-size:13px;margin:4px 0 0;\">'+subtitle+'</p>' if subtitle else ''}"
        f"</div>",
        unsafe_allow_html=True)

def divider():
    st.markdown(f"<div style='height:1px;background:{C['border']};margin:20px 0;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH PAGE
# ══════════════════════════════════════════════════════════════════════════════
if ACTIVE == "Search":
    section("Visual Search", "Find similar products by image, URL, product ID, or style fusion")

    qmode = st.radio("",
        ["📁  Upload Image", "🌐  Image URL", "🔢  Product ID", "🔀  Multi-Image Fusion"],
        horizontal=True, label_visibility="collapsed")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    uploaded = url_in = pid_in = multi = None

    if qmode == "📁  Upload Image":
        c1, c2 = st.columns([1, 3])
        with c1:
            uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
            if uploaded:
                st.image(Image.open(uploaded), use_container_width=True)
        with c2:
            if uploaded:
                st.markdown(f"""
                <div style='background:{C['card']};border:1px solid {C['border']};border-radius:10px;padding:16px;'>
                  <div style='color:{C['text']};font-weight:600;font-size:14px;margin-bottom:8px;'>Query Image</div>
                  <div style='color:{C['dim']};font-size:12px;'>📄 {uploaded.name}</div>
                  <div style='color:{C['dim']};font-size:12px;'>📦 {uploaded.size/1024:.1f} KB</div>
                  <div style='color:{C['dim']};font-size:12px;'>⚡ Method: {method.upper()}</div>
                  <div style='color:{C['dim']};font-size:12px;'>🎨 MMR: {"ON" if use_mmr else "OFF"}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:{C['card']};border:2px dashed {C['border']};border-radius:10px;
                padding:40px;text-align:center;'>
                  <div style='font-size:32px;margin-bottom:8px;'>📸</div>
                  <div style='color:{C['dim']};font-size:13px;'>Upload any fashion image</div>
                  <div style='color:{C['gray']};font-size:11px;margin-top:4px;'>JPG, JPEG, PNG</div>
                </div>
                """, unsafe_allow_html=True)

    elif qmode == "🌐  Image URL":
        url_in = st.text_input("", placeholder="https://example.com/jacket.jpg", label_visibility="collapsed")
        if url_in:
            try:
                r = requests.get(url_in, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
                c1, _ = st.columns([1, 3])
                with c1: st.image(Image.open(io.BytesIO(r.content)), use_container_width=True)
            except: st.info("Preview unavailable — will search anyway")

    elif qmode == "🔢  Product ID":
        c1, c2 = st.columns([1, 3])
        with c1:
            pid_in = st.number_input("", min_value=1, value=1163, step=1, label_visibility="collapsed")
        with c2:
            if st.button("👁  Preview Product", key="prev_btn"):
                try:
                    pm = requests.get(f"{API}/product/{int(pid_in)}", timeout=5).json()
                    ir = requests.get(f"{API}/image/{int(pid_in)}", timeout=5)
                    pc1, pc2 = st.columns([1, 3])
                    with pc1:
                        if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                    with pc2:
                        st.markdown(f"""
                        <div style='background:{C['card']};border:1px solid {C['border']};border-radius:10px;padding:14px;'>
                          <div style='color:{C['text']};font-weight:700;font-size:14px;margin-bottom:8px;'>{pm.get('name','')[:50]}</div>
                          <div style='color:{C['dim']};font-size:12px;'>Type: {pm.get('articleType','')}</div>
                          <div style='color:{C['dim']};font-size:12px;'>Color: {pm.get('baseColour','')}</div>
                          <div style='color:{C['gold']};font-size:12px;font-weight:600;'>🏷 {pm.get('brand','')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e: st.error(str(e))

    else:
        st.markdown(f"<div style='color:{C['dim']};font-size:13px;margin-bottom:10px;'>Upload 2–4 images — FashionFinder fuses their visual styles into one combined query</div>", unsafe_allow_html=True)
        multi = st.file_uploader("", type=["jpg","jpeg","png"], accept_multiple_files=True, label_visibility="collapsed")
        if multi and len(multi) >= 2:
            mc = st.columns(len(multi))
            for i, f in enumerate(multi):
                with mc[i]: st.image(Image.open(f), caption=f"Image {i+1}", use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    search_btn = st.button("🔍  Search Now", type="primary", use_container_width=True, key="search_main")

    if search_btn:
        p = {"k":k,"method":method,"use_mmr":use_mmr,"mmr_lambda":mmr_lambda}
        if cat_param:   p["category"] = cat_param
        if color_param: p["color"]    = color_param
        if brand_param: p["brand"]    = brand_param

        with st.spinner("Searching 44,419 products..."):
            try:
                if qmode == "📁  Upload Image" and uploaded:
                    uploaded.seek(0)
                    resp = requests.post(f"{API}/search/upload",
                        files={"file":(uploaded.name,uploaded.getvalue(),"image/jpeg")},
                        params=p, timeout=60).json()
                elif qmode == "🌐  Image URL" and url_in:
                    resp = requests.post(f"{API}/search/url", params={**p,"image_url":url_in}, timeout=60).json()
                elif qmode == "🔢  Product ID":
                    resp = requests.post(f"{API}/search/by_id",
                        params={**p,"product_id":int(pid_in)}, timeout=60).json()
                elif qmode == "🔀  Multi-Image Fusion" and multi and len(multi) >= 2:
                    fl = [("files",(f.name,f.getvalue(),"image/jpeg")) for f in multi]
                    resp = requests.post(f"{API}/search/multi_image", files=fl,
                        params={"k":k,"method":method,"use_mmr":True}, timeout=60).json()
                else:
                    st.warning("Please provide a query first."); st.stop()

                results = resp.get("results",[]); lat = resp.get("latency_ms",0)
                mlabels = {"faiss":"⚡ FAISS (ANN)","graph":"🕸️ k-NN Graph","baseline":"🎯 Baseline (Exact)"}
                mcolors = {"faiss":C['pink'],"graph":C['green'],"baseline":C['blue']}
                status(mlabels.get(method,method), lat, len(results), mcolors.get(method,C['blue']), use_mmr)

                g = st.columns(5)
                for i, item in enumerate(results[:10]): card(item, g[i%5])

            except Exception as e: st.error(f"Search failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# BRAND COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Brands":
    section("Cross-Brand Comparison", "Find the most similar item from each brand — no other platform does this")
    c1, c2 = st.columns([1, 2])
    with c1:
        bc_pid = st.number_input("Query Product ID", min_value=1, value=1163, step=1)
        bc_k   = st.slider("Results per brand", 1, 5, 2)
    with c2:
        sel_b = st.multiselect("Select brands to compare",
            options=all_brands,
            default=[b for b in ["Nike","Puma","Adidas","Reebok"] if b in all_brands][:4],
            max_selections=8)

    if st.button("⚡  Compare Brands", type="primary", use_container_width=True):
        if not sel_b: st.warning("Select at least one brand.")
        else:
            with st.spinner("Comparing across brands..."):
                try:
                    r = requests.get(f"{API}/brand_compare/{int(bc_pid)}",
                        params={"brands":",".join(sel_b),"k_per_brand":bc_k,"method":method},
                        timeout=60).json()
                    qp = r.get("query_product",{})
                    st.markdown(
                        f"<div style='background:{C['card']};border:1px solid {C['border']};border-radius:8px;"
                        f"padding:10px 16px;margin-bottom:16px;'>"
                        f"<span style='color:{C['dim']};font-size:11px;'>Query: </span>"
                        f"<span style='color:{C['text']};font-weight:600;'>{qp.get('name','')}</span>"
                        f"&nbsp;·&nbsp;<span style='color:{C['gold']};'>🏷 {qp.get('brand','')}</span>"
                        f"</div>", unsafe_allow_html=True)
                    divider()
                    bcols = st.columns(len(sel_b))
                    for i, brand in enumerate(sel_b):
                        with bcols[i]:
                            br = r.get("brand_results",{}).get(brand,{})
                            st.markdown(
                                f"<div style='background:{C['card']};border:1px solid {C['border']};border-radius:10px;"
                                f"padding:10px;margin-bottom:10px;text-align:center;'>"
                                f"<div style='color:{C['gold']};font-size:14px;font-weight:800;'>{brand}</div>"
                                f"<div style='color:{C['dim']};font-size:10px;'>{br.get('count',0):,} products · {br.get('latency_ms',0):.1f}ms</div>"
                                f"</div>", unsafe_allow_html=True)
                            for item in br.get("results",[]):
                                pid = item.get("id"); score = item.get("score",0)
                                try:
                                    ir = requests.get(f"{API}/image/{pid}", timeout=3)
                                    if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                                except: st.markdown("🖼️")
                                st.markdown(
                                    f"<div style='font-size:10px;color:{C['text']};'>{item.get('name','')[:24]}</div>"
                                    f"<div style='font-size:10px;color:{C['teal']};font-weight:600;'>{score:.3f}</div>"
                                    f"<div style='font-size:9px;color:{C['blue']};'>💡 {item.get('explanation','')[:40]}</div>"
                                    f"<hr style='border-color:{C['border']};margin:6px 0;'>",
                                    unsafe_allow_html=True)
                except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# STYLE TRANSFER
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Style":
    section("Style Transfer Search", "Same style, completely different color — unique to FashionFinder")
    c1, c2 = st.columns([1, 1])
    with c1:
        st_pid  = st.number_input("Product ID", min_value=1, value=1163, step=1)
        st_mode = st.radio("Mode", ["Single color", "All colors"])
    with c2:
        color_opts = fcolors if fcolors else ["Black","White","Red","Blue","Green","Navy Blue"]
        st_color = st.selectbox("Target Color", color_opts)
        st_k = st.slider("Results", 1, 10, 5)

    if st.button("🎨  Transfer Style", type="primary", use_container_width=True):
        with st.spinner("Searching..."):
            try:
                all_var = st_mode == "All colors"
                p = {"product_id":int(st_pid),"k":st_k,"all_variants":all_var}
                if not all_var: p["target_color"] = st_color
                if cat_param: p["category"] = cat_param
                r = requests.get(f"{API}/style_transfer", params=p, timeout=30).json()

                if "error" in r: st.error(r["error"])
                elif all_var:
                    variants = r.get("variants",{})
                    if not variants: st.warning("No variants found.")
                    else:
                        st.success(f"Found same style across {len(variants)} colors")
                        vlist = list(variants.items())
                        for rs in range(0, len(vlist), 5):
                            row = vlist[rs:rs+5]
                            rcols = st.columns(len(row))
                            for j,(cname,items) in enumerate(row):
                                with rcols[j]:
                                    st.markdown(f"<div style='text-align:center;color:{C['teal']};font-weight:700;font-size:11px;margin-bottom:4px;'>{cname}</div>", unsafe_allow_html=True)
                                    if items:
                                        pid = items[0].get("id")
                                        try:
                                            ir = requests.get(f"{API}/image/{pid}", timeout=3)
                                            if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                                        except: pass
                                        st.caption(f"{items[0].get('score',0):.3f}")
                else:
                    results = r.get("results",[]); query = r.get("query_product",{})
                    if not results: st.warning(f"No results in {st_color}.")
                    else:
                        cq, cr = st.columns([1, 5])
                        with cq:
                            try:
                                ir = requests.get(f"{API}/image/{query.get('id')}", timeout=3)
                                if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                            except: pass
                            st.caption(f"Original\n{query.get('baseColour','')}")
                        with cr:
                            rcols = st.columns(5)
                            for i, item in enumerate(results[:5]): card(item, rcols[i])
            except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# OUTFIT FINDER
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Outfit":
    section("Outfit Completion", "Given any product, get complementary items to complete the look")
    c1, c2 = st.columns([1, 3])
    with c1:
        of_pid = st.number_input("Product ID", min_value=1, value=1163, step=1)
        of_k   = st.slider("Suggestions per type", 1, 5, 2)
    if st.button("👔  Build Outfit", type="primary", use_container_width=True):
        with st.spinner("Building outfit..."):
            try:
                r = requests.get(f"{API}/outfit/{int(of_pid)}", params={"k":of_k}, timeout=30).json()
                qp = r.get("query_product",{}); sugg = r.get("suggestions",[]); types = r.get("complementary_types",[])
                cq, ci = st.columns([1, 3])
                with cq:
                    try:
                        ir = requests.get(f"{API}/image/{qp.get('id')}", timeout=3)
                        if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                    except: pass
                with ci:
                    st.markdown(
                        f"<div style='background:{C['card']};border:1px solid {C['border']};border-radius:10px;padding:16px;'>"
                        f"<div style='color:{C['text']};font-size:16px;font-weight:700;margin-bottom:8px;'>{qp.get('name','')[:50]}</div>"
                        f"<div style='color:{C['dim']};font-size:12px;'>Type: {qp.get('articleType','')} · Color: {qp.get('baseColour','')}</div>"
                        f"<div style='color:{C['gold']};font-size:12px;font-weight:600;margin-top:4px;'>🏷 {qp.get('brand','')}</div>"
                        f"<div style='margin-top:10px;color:{C['teal']};font-size:12px;'>Pairing with: <b>{', '.join(types)}</b></div>"
                        f"</div>", unsafe_allow_html=True)
                divider()
                for comp in types:
                    items = [s for s in sugg if s.get("complement_for") == comp]
                    if not items: continue
                    st.markdown(f"<div style='color:{C['text']};font-size:14px;font-weight:700;margin:12px 0 8px;'>👉 {comp}</div>", unsafe_allow_html=True)
                    ocols = st.columns(min(len(items),5))
                    for i, item in enumerate(items[:5]): card(item, ocols[i])
            except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUAL DNA
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "DNA":
    section("Visual DNA", "GradCAM attention map + 512-dim embedding heatmap — no other platform shows you this")
    dna_mode = st.radio("", ["Product ID", "Upload Image"], horizontal=True, label_visibility="collapsed")
    dna_pid_val = 0; dna_file = None
    if dna_mode == "Product ID":
        dna_pid_val = st.number_input("", min_value=1, value=1163, step=1, label_visibility="collapsed")
    else:
        dna_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if st.button("🧬  Analyze Visual DNA", type="primary", use_container_width=True):
        with st.spinner("Running GradCAM..."):
            try:
                if dna_mode == "Product ID":
                    r = requests.post(f"{API}/visual_dna", params={"product_id":int(dna_pid_val)}, timeout=60)
                else:
                    if not dna_file: st.warning("Upload an image first."); st.stop()
                    dna_file.seek(0)
                    r = requests.post(f"{API}/visual_dna",
                        files={"file":(dna_file.name,dna_file.getvalue(),"image/jpeg")}, timeout=60)
                if r.status_code != 200: st.error(f"Error: {r.status_code}"); st.stop()
                data = r.json()
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:8px;'>🔴 GradCAM Attention Overlay</div>", unsafe_allow_html=True)
                    st.caption("Red = ResNet18 paid most attention here")
                    pid_p = {"product_id":int(dna_pid_val)} if dna_mode=="Product ID" else {}
                    if dna_mode == "Product ID":
                        ov = requests.get(f"{API}/visual_dna/overlay", params=pid_p, timeout=30)
                    else:
                        dna_file.seek(0)
                        ov = requests.post(f"{API}/visual_dna/overlay_upload",
                            files={"file":(dna_file.name,dna_file.getvalue(),"image/jpeg")}, timeout=30)
                    if ov.status_code == 200: st.image(Image.open(io.BytesIO(ov.content)), use_container_width=True)
                    st.markdown(f"<div style='color:{C['text']};font-weight:600;font-size:12px;margin:8px 0 4px;'>Top attention regions:</div>", unsafe_allow_html=True)
                    for reg in data.get("top_regions",[]):
                        bar = int(reg.get("strength",0)*100)
                        st.markdown(
                            f"<div style='font-size:11px;color:{C['dim']};margin-bottom:4px;'>"
                            f"📍 {reg.get('region','')} — {reg.get('strength',0):.3f}"
                            f"<div style='background:{C['card']};border-radius:3px;height:3px;margin-top:2px;'>"
                            f"<div style='background:{C['pink']};width:{bar}%;height:3px;border-radius:3px;'></div></div></div>",
                            unsafe_allow_html=True)
                with dc2:
                    st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:8px;'>🟢 512-dim Embedding Heatmap</div>", unsafe_allow_html=True)
                    st.caption("Green = positive, Red = negative activation")
                    if dna_mode == "Product ID":
                        hm = requests.get(f"{API}/visual_dna/heatmap", params=pid_p, timeout=30)
                    else:
                        dna_file.seek(0)
                        hm = requests.post(f"{API}/visual_dna/heatmap_upload",
                            files={"file":(dna_file.name,dna_file.getvalue(),"image/jpeg")}, timeout=30)
                    if hm.status_code == 200: st.image(Image.open(io.BytesIO(hm.content)), use_container_width=True)
                    stats = data.get("embedding_stats",{})
                    if stats:
                        st.markdown(f"<div style='color:{C['text']};font-weight:600;font-size:12px;margin:8px 0 4px;'>Embedding stats:</div>", unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame([{
                            "Mean": round(stats.get("mean",0),4),
                            "Std":  round(stats.get("std",0),4),
                            "Max":  round(stats.get("max",0),4),
                            "L2":   round(stats.get("l2_norm",0),4),
                            "Non-zero": stats.get("nonzero_dims",0),
                        }]), use_container_width=True, hide_index=True)
            except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CLIP SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "CLIP":
    section("CLIP Style Search", "Upload image + pick a style modifier to transform results")
    if not health.get("clip_loaded",False):
        st.info("Install CLIP: `pip install git+https://github.com/openai/CLIP.git` then restart API")

    c1, c2 = st.columns([1, 2])
    with c1:
        clip_img = st.file_uploader("Upload query image", type=["jpg","jpeg","png"])
        if clip_img:
            st.image(Image.open(clip_img), use_container_width=True)

    with c2:
        st.markdown(f"<div style='color:{C['text']};font-weight:600;font-size:13px;margin-bottom:10px;'>Select style modifier:</div>", unsafe_allow_html=True)
        STYLES = ["formal","casual","sporty","elegant","party","minimal","colorful","vintage","winter"]
        style_cols = st.columns(3)
        for i, s in enumerate(STYLES):
            with style_cols[i % 3]:
                is_sel = st.session_state.clip_style == s
                if st.button(
                    ("✅ " if is_sel else "") + s.title(),
                    key=f"cs_{i}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary"
                ):
                    st.session_state.clip_style = s
                    st.rerun()

        if st.session_state.clip_style:
            st.markdown(
                f"<div style='background:#091828;border:1px solid {C['blue']};border-radius:8px;"
                f"padding:8px 12px;margin-top:10px;'>"
                f"<span style='color:{C['dim']};font-size:11px;'>Active: </span>"
                f"<span style='color:{C['blue']};font-weight:700;'>{st.session_state.clip_style}</span>"
                f"</div>", unsafe_allow_html=True)

        custom = st.text_input("Or type custom style", placeholder="beach outfit, festive wear...", key="clip_custom")
        clip_k = st.slider("Results", 1, 20, 10, key="clip_k")
        img_w  = st.slider("Image weight", 0.0, 1.0, 0.7, 0.05)

    if st.button("🤖  CLIP Style Search", type="primary", use_container_width=True,
                  disabled=not health.get("clip_loaded",False)):
        current_style = custom.strip() or st.session_state.clip_style
        with st.spinner("Running style search..."):
            try:
                p = {"k":clip_k,"method":method,"image_weight":img_w,"text_weight":round(1-img_w,2)}
                if cat_param: p["category"] = cat_param
                if clip_img and current_style:
                    clip_img.seek(0)
                    r = requests.post(f"{API}/clip_search",
                        files={"file":(clip_img.name,clip_img.getvalue(),"image/jpeg")},
                        params={**p,"text":current_style}, timeout=60).json()
                elif clip_img:
                    clip_img.seek(0)
                    r = requests.post(f"{API}/clip_search",
                        files={"file":(clip_img.name,clip_img.getvalue(),"image/jpeg")},
                        params=p, timeout=60).json()
                elif current_style:
                    r = requests.post(f"{API}/clip_search", params={**p,"text":current_style}, timeout=60).json()
                else:
                    st.warning("Upload image or select style."); st.stop()
                results = r.get("results",[]); lat = r.get("latency_ms",0)
                status(f"🤖 CLIP ({r.get('mode','')})", lat, len(results), C['purple'])
                g = st.columns(5)
                for i, item in enumerate(results[:10]): card(item, g[i%5])
            except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ADD PRODUCT
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Add":
    section("Real-Time Product Index", "Add any image — immediately searchable without rebuilding the index")
    c1, c2 = st.columns([1, 1])
    with c1:
        new_img = st.file_uploader("Product Image", type=["jpg","jpeg","png"])
        if new_img: st.image(Image.open(new_img), use_container_width=True)
    with c2:
        new_name  = st.text_input("Product Name", value="My Custom Product")
        new_cat   = st.selectbox("Category", categories or ["Apparel"])
        new_color = st.selectbox("Color", fcolors or ["Black","White","Blue"])
        new_brand = st.text_input("Brand", value="Custom")
        new_type  = st.text_input("Article Type", value="Tshirts")

    if st.button("➕  Add to Index", type="primary", use_container_width=True):
        if not new_img: st.warning("Upload image first.")
        elif not new_name.strip(): st.warning("Enter product name.")
        else:
            with st.spinner("Embedding and indexing..."):
                try:
                    new_img.seek(0)
                    r = requests.post(f"{API}/realtime/add",
                        files={"file":(new_img.name,new_img.getvalue(),"image/jpeg")},
                        params={"name":new_name,"category":new_cat,"color":new_color,
                                "brand":new_brand,"article_type":new_type}, timeout=30).json()
                    st.success(f"✅ Added! Product ID: **{r.get('id','?')}** — Immediately searchable!")
                    st.json(r)
                except Exception as e: st.error(f"Failed: {e}")

    divider()
    st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:12px;'>🔴 Live Buffer</div>", unsafe_allow_html=True)
    try:
        rt_data = requests.get(f"{API}/realtime/products", timeout=5).json()
        rt_prods = rt_data.get("products",[])
        if rt_prods:
            rtcols = st.columns(5)
            for i,p in enumerate(rt_prods[:10]):
                with rtcols[i%5]:
                    try:
                        ir = requests.get(f"{API}/image/{p.get('id')}", timeout=2)
                        if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                    except: st.markdown("🖼️")
                    st.caption(f"🔴 {p.get('productDisplayName','')[:20]}")
            if st.button("🗑️ Clear Buffer"):
                requests.delete(f"{API}/realtime/clear", timeout=5)
                st.success("Cleared."); st.rerun()
        else: st.markdown(f"<div style='color:{C['dim']};font-size:13px;'>No products in buffer yet.</div>", unsafe_allow_html=True)
    except: pass

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Benchmark":
    section("Live Benchmark", "Real latency and recall numbers — every result is live and reproducible")
    bench_n = st.number_input("Number of queries", 10, 500, 50, step=10)
    if st.button("▶️  Run Benchmark", type="primary"):
        with st.spinner(f"Running {bench_n} queries across all 3 methods..."):
            try:
                bd = requests.post(f"{API}/benchmark", params={"n":bench_n,"k":k}, timeout=300).json()
                st.session_state["bench_data"] = bd
            except Exception as e: st.error(f"Failed: {e}")

    if "bench_data" in st.session_state:
        bd = st.session_state["bench_data"]
        mc = st.columns(6)
        for col, label, val, color in [
            (mc[0],"Baseline Median",f"{bd['baseline_median_ms']} ms",C['blue']),
            (mc[1],"Graph Median",   f"{bd['graph_median_ms']} ms",   C['green']),
            (mc[2],"FAISS Median",   f"{bd['faiss_median_ms']} ms",   C['pink']),
            (mc[3],"Baseline p95",   f"{bd['baseline_p95_ms']} ms",   C['blue']),
            (mc[4],"Graph p95",      f"{bd['graph_p95_ms']} ms",      C['green']),
            (mc[5],"FAISS p95",      f"{bd['faiss_p95_ms']} ms",      C['pink']),
        ]:
            with col:
                st.markdown(
                    f"<div style='background:{C['card']};border:1px solid {color}33;border-top:2px solid {color};"
                    f"border-radius:8px;padding:12px;text-align:center;'>"
                    f"<div style='color:{C['dim']};font-size:10px;font-weight:600;text-transform:uppercase;margin-bottom:4px;'>{label}</div>"
                    f"<div style='color:{color};font-size:20px;font-weight:800;'>{val}</div>"
                    f"</div>", unsafe_allow_html=True)

        divider()
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:8px;'>Latency (ms)</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5))
            fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0f1a2e")
            methods = ["Baseline","Graph","FAISS"]
            medians = [bd["baseline_median_ms"],bd["graph_median_ms"],bd["faiss_median_ms"]]
            p95s    = [bd["baseline_p95_ms"],bd["graph_p95_ms"],bd["faiss_p95_ms"]]
            x = np.arange(3); w = 0.35
            ax.bar(x-w/2,medians,w,color=["#4FC3F7","#4ade80","#f472b6"],alpha=0.9,label="Median")
            ax.bar(x+w/2,p95s,  w,color=["#4FC3F7","#4ade80","#f472b6"],alpha=0.45,label="p95")
            ax.set_xticks(x); ax.set_xticklabels(methods,color="#94a3b8",fontsize=10)
            ax.set_ylabel("ms",color="#94a3b8"); ax.tick_params(colors="#64748b")
            ax.legend(facecolor="#0f1a2e",labelcolor="#94a3b8",framealpha=0.8)
            ax.spines[:].set_color("#1e3a5f"); ax.grid(axis='y',color="#1e3a5f",alpha=0.5)
            for bar in list(ax.patches[:3]):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                        f"{bar.get_height():.1f}",ha="center",va="bottom",color="white",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with bc2:
            st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:8px;'>Recall@k</div>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5,3.5))
            fig2.patch.set_facecolor("#080c14"); ax2.set_facecolor("#0f1a2e")
            recalls = [1.000,bd["graph_recall"],bd["faiss_recall"]]
            bars = ax2.bar(methods,recalls,color=["#4FC3F7","#4ade80","#f472b6"],alpha=0.9,width=0.5)
            ax2.set_ylim(0,1.1); ax2.set_ylabel("Recall",color="#94a3b8")
            ax2.tick_params(colors="#64748b"); ax2.spines[:].set_color("#1e3a5f")
            ax2.set_xticklabels(methods,color="#94a3b8",fontsize=10)
            ax2.axhline(y=1.0,color="#2a3550",linestyle="--",linewidth=1)
            ax2.grid(axis='y',color="#1e3a5f",alpha=0.5)
            for bar,val in zip(bars,recalls):
                ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                         f"{val:.3f}",ha="center",va="bottom",color="white",fontsize=11,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.dataframe(pd.DataFrame({
            "Method":["Baseline","Graph","FAISS"],
            "Median (ms)":[bd["baseline_median_ms"],bd["graph_median_ms"],bd["faiss_median_ms"]],
            "p95 (ms)":[bd["baseline_p95_ms"],bd["graph_p95_ms"],bd["faiss_p95_ms"]],
            "Recall@k":[1.000,bd["graph_recall"],bd["faiss_recall"]],
            "Type":["Exact","Approximate","Approximate"],
        }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Graph":
    section("k-NN Similarity Graph", "Visualize product relationships — no other platform shows you this")
    c1, _ = st.columns([1, 2])
    with c1:
        ge_pid  = st.number_input("Center Product ID", min_value=1, value=1163, step=1)
        ge_hops = st.slider("Hops", 1, 3, 2)
    if st.button("🕸️  Show Graph", type="primary"):
        with st.spinner("Building graph..."):
            try:
                gdata = requests.get(f"{API}/graph_neighbors/{int(ge_pid)}",
                    params={"hops":ge_hops}, timeout=15).json()
                G = nx.Graph(); cn = gdata["center"]["id"]; G.add_node(cn)
                CMAP = {"Apparel":"#4ade80","Footwear":"#fb923c","Accessories":"#f472b6",
                        "Personal Care":"#a78bfa","Sporting Goods":"#2dd4bf","Free Items":"#fbbf24","Home":"#94a3b8"}
                node_labels = {cn:f"#{cn}\nQUERY"}
                for nd in gdata.get("nodes",[]):
                    nid = nd["id"]; G.add_node(nid)
                    G.add_edge(nd["parent"],nid,weight=nd.get("score",0))
                    node_labels[nid] = f"{nd.get('articleType','')[:8]}\n{nd.get('score',0):.2f}"
                n_colors=[]; n_sizes=[]
                for n in G.nodes():
                    if n == cn: n_colors.append("#fbbf24"); n_sizes.append(1200)
                    else:
                        cat = next((nd.get("masterCategory","?") for nd in gdata.get("nodes",[]) if nd["id"]==n),"?")
                        n_colors.append(CMAP.get(cat,"#64748b")); n_sizes.append(400)
                fig,ax = plt.subplots(figsize=(12,7))
                fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#080c14")
                pos = nx.spring_layout(G,seed=42,k=1.5)
                nx.draw_networkx_nodes(G,pos,node_color=n_colors,node_size=n_sizes,ax=ax,alpha=0.9)
                nx.draw_networkx_edges(G,pos,edge_color="#1e3a5f",ax=ax,alpha=0.7,width=1.2)
                nx.draw_networkx_labels(G,pos,labels=node_labels,font_size=6,font_color="white",ax=ax)
                patches = [mpatches.Patch(color=c,label=cat) for cat,c in CMAP.items()]
                patches.append(mpatches.Patch(color="#fbbf24",label="Query"))
                ax.legend(handles=patches,loc="upper left",facecolor="#0f1a2e",
                         labelcolor="white",fontsize=8,framealpha=0.9,edgecolor="#1e3a5f")
                ax.axis("off")
                ax.set_title(f"k-NN Graph: Product {ge_pid} · {ge_hops} hops · {G.number_of_nodes()} nodes",
                            color=C['blue'],fontsize=13,fontweight="bold",pad=15)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                rows = [{"ID":nd["id"],"Name":nd.get("name","")[:40],"Brand":nd.get("brand",""),
                         "Category":nd.get("masterCategory",""),"Score":nd.get("score",0),"Depth":nd.get("depth",1)}
                        for nd in gdata.get("nodes",[])]
                if rows:
                    divider()
                    st.dataframe(pd.DataFrame(rows).sort_values("Score",ascending=False),
                                use_container_width=True, hide_index=True)
            except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# FASHION TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Timeline":
    section("Fashion Timeline", "How colors, styles, and volumes evolved year by year")
    if st.button("📅  Load Timeline", type="primary"):
        with st.spinner("Analyzing trends..."):
            try:
                tl = requests.get(f"{API}/timeline", timeout=30).json()
                years = tl.get("years",[])
                if not years: st.warning("No year data."); st.stop()
                st.success(f"Data for {len(years)} years: {years[0]} → {years[-1]}")
                st.markdown(f"<div style='color:{C['text']};font-weight:600;margin:16px 0 8px;'>Product Volume by Year</div>", unsafe_allow_html=True)
                vr = requests.get(f"{API}/timeline/volume_chart", timeout=20)
                if vr.status_code == 200: st.image(Image.open(io.BytesIO(vr.content)), use_container_width=True)
                divider()
                tl1,tl2 = st.columns(2)
                with tl1:
                    st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:8px;'>Color Trends</div>", unsafe_allow_html=True)
                    cr = requests.get(f"{API}/timeline/color_chart", timeout=20)
                    if cr.status_code == 200: st.image(Image.open(io.BytesIO(cr.content)), use_container_width=True)
                with tl2:
                    st.markdown(f"<div style='color:{C['text']};font-weight:600;margin-bottom:8px;'>Article Type Trends</div>", unsafe_allow_html=True)
                    ar = requests.get(f"{API}/timeline/article_chart", timeout=20)
                    if ar.status_code == 200: st.image(Image.open(io.BytesIO(ar.content)), use_container_width=True)
                divider()
                sel_yr = st.select_slider("Explore year", options=years, value=years[-1])
                yr = requests.get(f"{API}/timeline/year/{sel_yr}", timeout=10).json()
                if "error" not in yr:
                    yc1,yc2,yc3 = st.columns(3)
                    with yc1:
                        st.markdown(f"<div style='color:{C['teal']};font-weight:700;margin-bottom:8px;'>{sel_yr} — Colors</div>", unsafe_allow_html=True)
                        for item in yr.get("top_colors",[]): st.markdown(f"<div style='color:{C['dim']};font-size:12px;'>🎨 <b style='color:{C['text']};'>{item['color']}</b>: {item['count']:,}</div>", unsafe_allow_html=True)
                    with yc2:
                        st.markdown(f"<div style='color:{C['teal']};font-weight:700;margin-bottom:8px;'>{sel_yr} — Types</div>", unsafe_allow_html=True)
                        for item in yr.get("top_articles",[]): st.markdown(f"<div style='color:{C['dim']};font-size:12px;'>👕 <b style='color:{C['text']};'>{item['type']}</b>: {item['count']:,}</div>", unsafe_allow_html=True)
                    with yc3:
                        st.markdown(f"<div style='color:{C['teal']};font-weight:700;margin-bottom:8px;'>{sel_yr} — Categories</div>", unsafe_allow_html=True)
                        for item in yr.get("top_categories",[]): st.markdown(f"<div style='color:{C['dim']};font-size:12px;'>📁 <b style='color:{C['text']};'>{item['category']}</b>: {item['count']:,}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:12px;background:{C['card']};border:1px solid {C['border']};border-radius:8px;padding:10px 16px;color:{C['blue']};font-weight:600;'>Total products in {sel_yr}: {yr.get('total_products',0):,}</div>", unsafe_allow_html=True)
            except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TRENDING
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Trending":
    section("Trending Now", "Live search trends — updates with every search")
    hours = st.slider("Time window (hours)", 1, 168, 24)
    try:
        td = requests.get(f"{API}/trending", params={"hours":hours,"top_n":20}, timeout=10).json()
        total = td.get("total_searches",0)
        st.markdown(f"<div style='color:{C['dim']};font-size:13px;margin-bottom:16px;'>{total} total searches in the last {hours} hours</div>", unsafe_allow_html=True)
        divider()
        tc1,tc2,tc3 = st.columns(3)
        with tc1:
            st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:12px;'>🔥 Trending Products</div>", unsafe_allow_html=True)
            prods = td.get("trending_products",[])
            if prods:
                for p in prods[:8]:
                    pid=p.get("id"); views=p.get("views",0)
                    ct,ci = st.columns([3,1])
                    with ct:
                        st.markdown(f"<div style='color:{C['text']};font-size:12px;font-weight:600;'>{p.get('name','')[:28]}</div><div style='color:{C['gold']};font-size:11px;'>🏷 {p.get('brand','')} &nbsp;·&nbsp; 👁 {views}</div>", unsafe_allow_html=True)
                    with ci:
                        try:
                            ir = requests.get(f"{API}/image/{pid}", timeout=2)
                            if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), width=55)
                        except: pass
                    st.markdown(f"<div style='height:1px;background:{C['border']};margin:6px 0;'></div>", unsafe_allow_html=True)
            else: st.markdown(f"<div style='color:{C['dim']};font-size:13px;'>Do some searches to see trends!</div>", unsafe_allow_html=True)
        with tc2:
            st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:12px;'>📂 Categories</div>", unsafe_allow_html=True)
            cats = td.get("trending_categories",[])
            if cats:
                fig,ax = plt.subplots(figsize=(4,3)); fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0f1a2e")
                ax.barh([c["category"] for c in cats[:7]],[c["searches"] for c in cats[:7]],color="#4FC3F7",alpha=0.85)
                ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e3a5f"); ax.set_xlabel("Searches",color="#94a3b8")
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with tc3:
            st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:12px;'>🏷️ Brands</div>", unsafe_allow_html=True)
            bts = td.get("trending_brands",[])
            if bts:
                fig,ax = plt.subplots(figsize=(4,3)); fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0f1a2e")
                ax.barh([b["brand"] for b in bts[:7]],[b["searches"] for b in bts[:7]],color="#fbbf24",alpha=0.85)
                ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e3a5f"); ax.set_xlabel("Searches",color="#94a3b8")
                plt.tight_layout(); st.pyplot(fig); plt.close()
        divider()
        hist = requests.get(f"{API}/search_history",params={"limit":20},timeout=5).json().get("history",[])
        if hist:
            st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:8px;'>📋 Recent Searches</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
    except Exception as e: st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif ACTIVE == "Assistant":
    section("AI Fashion Assistant", "Chat to find products, build outfits, get style advice — powered by RAG + Claude")

    if "chatbot_instance" not in st.session_state:
        with st.spinner("Loading AI assistant..."):
            try:
                from chatbot import RAGFashionChatbot
                from heap_ranker import top_k_cosine as tkc
                emb  = np.load("embeddings/embeddings.npy")
                meta = pd.read_csv("embeddings/metadata.csv").reset_index(drop=True)
                from embedder import Embedder
                from hash_index import HashIndex
                from color_index import ColorIndex
                from brand_index import BrandIndex
                from faiss_index import FAISSIndex
                hi = HashIndex(meta); ci = ColorIndex(meta); bi = BrandIndex(meta)
                fi = FAISSIndex()
                if os.path.exists("embeddings/faiss.index"): fi.load("embeddings/faiss.index")
                st.session_state["chatbot_instance"] = RAGFashionChatbot(
                    metadata=meta, embeddings=emb,
                    hash_index=hi, color_index=ci, brand_index=bi,
                    heap_ranker_fn=tkc, embedder=Embedder(), faiss_index=fi,
                    api_key=os.environ.get("ANTHROPIC_API_KEY",""))
            except Exception as e:
                st.error(f"Could not load chatbot: {e}"); st.stop()

    bot = st.session_state["chatbot_instance"]

    # Quick questions
    st.markdown(f"<div style='color:{C['dim']};font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>Quick Questions</div>", unsafe_allow_html=True)
    qq = st.columns(4)
    QUICK = [
        ("👰 Wedding","What should I wear to a wedding? Traditional Indian outfit"),
        ("💪 Gym","Build me a complete gym workout outfit"),
        ("💼 Office","What is a good professional office look?"),
        ("🎉 Party","Suggest a party outfit for a night out"),
        ("🏖️ Beach","What should I wear to the beach?"),
        ("❄️ Winter","Build me a cozy winter outfit"),
        ("🎓 College","Good college outfit for everyday wear?"),
        ("💕 Date Night","Help me pick a romantic date night outfit"),
    ]
    for i,(label,question) in enumerate(QUICK):
        with qq[i%4]:
            if st.button(label, key=f"qq_{i}", use_container_width=True):
                st.session_state["pending_q"] = question; st.rerun()

    if "pending_q" in st.session_state:
        q = st.session_state.pop("pending_q")
        with st.spinner("Thinking..."):
            resp, prods, tts = bot.chat(q)
            st.session_state.chat_history.append(("user", q))
            st.session_state.chat_history.append(("assistant", resp))
            st.session_state.chat_products = prods
            st.session_state["tts_js"] = tts
        st.rerun()

    divider()
    chat_col, prod_col = st.columns([3, 2])

    with chat_col:
        # Chat history
        chat_box = st.container()
        with chat_box:
            if not st.session_state.chat_history:
                st.markdown(
                    f"<div style='background:{C['card']};border:1px dashed {C['border']};border-radius:12px;"
                    f"padding:40px;text-align:center;'>"
                    f"<div style='font-size:32px;margin-bottom:8px;'>💬</div>"
                    f"<div style='color:{C['text']};font-size:16px;font-weight:600;margin-bottom:8px;'>Start a conversation</div>"
                    f"<div style='color:{C['dim']};font-size:13px;'>Ask about outfits, find products, or get style advice</div>"
                    f"<div style='color:{C['dim']};font-size:12px;margin-top:12px;'>"
                    f"Try: <i>'What to wear to a wedding?'</i> or <i>'Show me red Nike shoes'</i>"
                    f"</div></div>",
                    unsafe_allow_html=True)
            else:
                for role, msg in st.session_state.chat_history:
                    if role == "user":
                        st.markdown(
                            f"<div style='display:flex;justify-content:flex-end;margin:8px 0;'>"
                            f"<div style='background:#1a3a5c;border:1px solid #2a5a8c;border-radius:14px 14px 2px 14px;"
                            f"padding:10px 14px;max-width:80%;'>"
                            f"<div style='color:{C['text']};font-size:13px;'>{msg}</div>"
                            f"</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div style='display:flex;justify-content:flex-start;margin:8px 0;gap:8px;'>"
                            f"<div style='width:28px;height:28px;background:linear-gradient(135deg,#0066cc,#004499);"
                            f"border-radius:50%;display:flex;align-items:center;justify-content:center;"
                            f"font-size:14px;flex-shrink:0;margin-top:2px;'>👗</div>"
                            f"<div style='background:{C['card']};border:1px solid {C['border']};border-radius:2px 14px 14px 14px;"
                            f"padding:10px 14px;max-width:80%;'>"
                            f"<div style='color:{C['teal']};font-size:10px;font-weight:700;margin-bottom:4px;'>FASHIONFINDER AI</div>"
                            f"<div style='color:{C['text']};font-size:13px;line-height:1.5;'>{msg}</div>"
                            f"</div></div>", unsafe_allow_html=True)

        # Input area
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        user_input = st.text_input("", placeholder="Ask me anything about fashion...",
                                    key="chat_in", label_visibility="collapsed")
        c1,c2,c3 = st.columns([4,1,1])
        with c1: send = st.button("💬  Send", type="primary", use_container_width=True, key="chat_send")
        with c2: tts_on = st.toggle("🔊", value=True, key="tts_tog", help="Speak response")
        with c3: reset = st.button("🔄", use_container_width=True, key="chat_rst", help="Reset")

        if send and user_input.strip():
            with st.spinner("Thinking..."):
                resp, prods, tts = bot.chat(user_input)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", resp))
                st.session_state.chat_products = prods
                if tts_on: st.session_state["tts_js"] = tts
            st.rerun()

        if reset:
            bot.reset()
            st.session_state.chat_history = []
            st.session_state.chat_products = []
            st.session_state.pop("tts_js", None)
            st.rerun()

        # TTS injection
        if tts_on and st.session_state.get("tts_js"):
            import streamlit.components.v1 as components
            components.html(st.session_state.pop("tts_js",""), height=0)

        # Follow-up suggestions
        if st.session_state.chat_history:
            st.markdown(f"<div style='color:{C['dim']};font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-top:16px;margin-bottom:8px;'>Suggested follow-ups</div>", unsafe_allow_html=True)
            sugs = ["Show me something more colorful","What shoes go with this?",
                    "Find me a similar item from Nike","What accessories match?","Make it more formal"]
            sc = st.columns(5)
            for i,s in enumerate(sugs):
                with sc[i]:
                    if st.button(s[:20], key=f"sug_{i}", use_container_width=True):
                        st.session_state["pending_q"] = s; st.rerun()

    with prod_col:
        st.markdown(f"<div style='color:{C['text']};font-weight:700;margin-bottom:12px;'>Products from Chat</div>", unsafe_allow_html=True)
        chat_prods = st.session_state.get("chat_products",[])
        if chat_prods:
            for p in chat_prods:
                pid = p.get("id")
                pc1,pc2 = st.columns([1,2])
                with pc1:
                    try:
                        ir = requests.get(f"{API}/image/{pid}", timeout=2)
                        if ir.status_code == 200: st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                    except: st.markdown("🖼️")
                with pc2:
                    role = p.get("outfit_role","")
                    st.markdown(
                        f"{'<div style=\"color:'+C['teal']+';font-size:10px;font-weight:700;margin-bottom:2px;\">'+role+'</div>' if role else ''}"
                        f"<div style='color:{C['text']};font-size:11px;font-weight:600;'>{p.get('name','')[:28]}</div>"
                        f"<div style='color:{C['dim']};font-size:10px;'>{p.get('type','')} · {p.get('color','')}</div>"
                        f"<div style='color:{C['gold']};font-size:10px;'>🏷 {p.get('brand','')}</div>"
                        f"<div style='color:{C['blue']};font-size:10px;'>ID: {pid}</div>",
                        unsafe_allow_html=True)
                st.markdown(f"<div style='height:1px;background:{C['border']};margin:8px 0;'></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:{C['card']};border:1px dashed {C['border']};border-radius:10px;"
                f"padding:30px;text-align:center;color:{C['dim']};font-size:13px;'>"
                f"Products mentioned in chat will appear here</div>",
                unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="height:1px;background:linear-gradient(90deg,transparent,{C['border']},transparent);margin:32px 0 16px;"></div>
<div style="text-align:center;padding-bottom:16px;">
  <span style="color:{C['dim']};font-size:11px;letter-spacing:1px;">
    FASHIONFINDER v4.0 &nbsp;·&nbsp;
    <span style="color:{C['blue']};">Akila Lourdes Miriyala Francis</span>
    &nbsp;&amp;&nbsp;
    <span style="color:{C['green']};">Akilan Manivannan</span>
    &nbsp;·&nbsp; LIU Brooklyn &nbsp;·&nbsp; 2026
  </span>
</div>
""", unsafe_allow_html=True)
