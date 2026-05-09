"""
ui.py - FashionFinder v4.0 Ultimate
Tab state fixed: uses session_state navigation so buttons never reset the tab
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

st.set_page_config(page_title="FashionFinder v4", page_icon="👗", layout="wide")

st.markdown("""
<style>
.stApp { background: #0a0e1a; }
/* Prevent nav buttons from wrapping */
div[data-testid="column"] button {
    font-size: 11px !important;
    padding: 4px 2px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.metric-box { background:#12192b; border:1px solid #2a3550; border-radius:10px; padding:12px 8px; text-align:center; }
.status-bar { background:linear-gradient(90deg,#0d2137,#0d3730); border:1px solid #1e4d3b; border-radius:8px; padding:8px 16px; margin-bottom:12px; }
.expl { background:#0d2137; border-left:3px solid #4FC3F7; border-radius:4px; padding:4px 8px; font-size:11px; color:#81C784; margin-top:3px; }
.brand-pill { display:inline-block; background:linear-gradient(90deg,#7B5E00,#FFD700); color:#000; border-radius:10px; padding:2px 8px; font-size:10px; font-weight:bold; }
.section-title { font-size:22px; font-weight:800; color:#4FC3F7; margin-bottom:4px; }
.section-sub { font-size:13px; color:#8B949E; margin-bottom:14px; }
[data-testid="stSidebar"] { background:#0d1120 !important; }
div.stButton > button { transition: all 0.2s; }
</style>
""", unsafe_allow_html=True)

BLUE="#4FC3F7"; GREEN="#81C784"; PINK="#F48FB1"
PURPLE="#CE93D8"; GOLD="#FFD700"; ORANGE="#FFB74D"
GRAY="#8B949E"; TEAL="#80DEEA"; RED="#EF5350"

# ── Session state init ─────────────────────────────────────────────────────────
if "tab" not in st.session_state:
    st.session_state.tab = "Search"

TAB_NAMES = [
    "🔍 Search", "🏷️ Brands", "🎨 Style",
    "👔 Outfit", "🧬 DNA", "🤖 CLIP",
    "➕ Add", "📊 Benchmark", "🕸️ Graph",
    "📅 Timeline", "🔥 Trending", "💬 Assistant"
]

# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def api_get(endpoint, **params):
    try: return requests.get(f"{API}/{endpoint}", params=params, timeout=5).json()
    except: return {}

health     = api_get("health")
categories = [c["name"] for c in api_get("categories").get("categories", [])]
all_brands = [b["name"] for b in api_get("brands", top_n=150).get("brands", [])]
all_colors = api_get("colors").get("colors", [])

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:16px 0 6px;">
  <span style="font-size:50px;font-weight:900;
    background:linear-gradient(90deg,{BLUE},{TEAL},{GREEN});
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    FashionFinder
  </span>
  <span style="font-size:15px;color:{GOLD};font-weight:700;margin-left:8px;">v4.0 Ultimate</span>
  <br>
  <span style="font-size:11px;color:#444;letter-spacing:1px;">
    RESNET18 · k-NN GRAPH · FAISS ANN · MMR DIVERSITY · BRAND COMPARE ·
    STYLE TRANSFER · VISUAL DNA · CLIP · REAL-TIME INDEX · FASHION TIMELINE
  </span>
</div>
<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,{BLUE},transparent);margin:0 0 10px;">
""", unsafe_allow_html=True)

# ── Navigation bar (session_state based - NO TAB RESET) ────────────────────────
# Navigation - compact single row, no wrapping
nav_cols = st.columns(len(TAB_NAMES))
for i, (col, name) in enumerate(zip(nav_cols, TAB_NAMES)):
    with col:
        is_active = st.session_state.tab == name
        # Use just emoji + very short label to prevent wrapping
        short = name  # already short from TAB_NAMES
        if st.button(short, key=f"nav_{i}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.tab = name
            st.rerun()

st.markdown("<hr style='border:none;height:1px;background:#2a3550;margin:4px 0 14px;'>", unsafe_allow_html=True)

ACTIVE = st.session_state.tab

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{BLUE};font-weight:800;font-size:15px;'>⚙️ Settings</div>", unsafe_allow_html=True)
    method = st.radio("Retrieval Method", ["faiss","graph","baseline"],
                       format_func=lambda x: {"faiss":"⚡ FAISS (Fastest)","graph":"🕸️ Graph (Scalable)","baseline":"🎯 Baseline (Exact)"}[x])
    k = st.slider("Top-k Results", 1, 20, 10)
    use_mmr = st.toggle("🎨 MMR Diversity Reranking")
    mmr_lambda = st.slider("Diversity ←→ Relevance", 0.0, 1.0, 0.6, 0.05) if use_mmr else 0.6

    st.markdown("---")
    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:13px;'>🔍 Filters</div>", unsafe_allow_html=True)
    sel_cat     = st.selectbox("Category", ["All"] + categories)
    cat_param   = None if sel_cat == "All" else sel_cat
    filt_colors = api_get("colors", category=cat_param).get("colors", all_colors) if cat_param else all_colors
    sel_color   = st.selectbox("Color", ["All"] + filt_colors)
    color_param = None if sel_color == "All" else sel_color
    sel_brand   = st.selectbox("Brand", ["All"] + all_brands)
    brand_param = None if sel_brand == "All" else sel_brand

    st.markdown("---")
    st.markdown(f"<div style='color:{GREEN};font-weight:700;font-size:13px;'>📊 System Status</div>", unsafe_allow_html=True)
    sc = st.columns(3)
    for col, label, val, color in [
        (sc[0],"Products",f"{health.get('products_indexed',0):,}",BLUE),
        (sc[1],"Brands",  f"{health.get('brands_detected',0):,}", GOLD),
        (sc[2],"Searches",f"{health.get('total_searches',0):,}",  GREEN),
    ]:
        with col:
            st.markdown(f"<div class='metric-box'><div style='color:#aaa;font-size:9px;'>{label}</div>"
                        f"<div style='color:{color};font-size:15px;font-weight:800;'>{val}</div></div>",
                        unsafe_allow_html=True)
    g=health.get("graph_loaded",False); f=health.get("faiss_loaded",False); c=health.get("clip_loaded",False)
    st.markdown(f"<div style='font-size:11px;margin-top:6px;'>Graph {'✅' if g else '❌'} | FAISS {'✅' if f else '❌'} | CLIP {'✅' if c else '⚠️'}</div>", unsafe_allow_html=True)
    rt = health.get("realtime_buffer_size",0)
    if rt>0: st.markdown(f"<div style='color:{RED};font-size:11px;'>🔴 {rt} live products</div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🧩 Data Structures"):
        for ds,fl,role,color in [
            ("Hash Table","hash_index.py","Category→indices O(1)",BLUE),
            ("Nested Hash","color_index.py","Cat+Color→idx O(1)",GREEN),
            ("Brand Index","brand_index.py","467 brands",GOLD),
            ("k-NN Graph","knn_graph.py","444K edges BFS",PINK),
            ("Min-Heap","heap_ranker.py","Top-k O(n log k)",PURPLE),
            ("FAISS IVF","faiss_index.py","100 clusters ANN",TEAL),
        ]:
            st.markdown(f"<div style='margin-bottom:5px;'><span style='color:{color};font-weight:bold;font-size:11px;'>{ds}</span><br>"
                        f"<span style='color:#555;font-size:10px;'>{fl} — {role}</span></div>", unsafe_allow_html=True)

# ── Card renderer ──────────────────────────────────────────────────────────────
def render_card(item, col, show_exp=True):
    with col:
        pid=item.get("id"); score=item.get("score",0); name=item.get("name","")[:26]
        atype=item.get("articleType",""); clr=item.get("baseColour",""); brand=item.get("brand","")
        expl=item.get("explanation",""); rank=item.get("rank",""); is_rt=item.get("is_realtime",False)
        try:
            r=requests.get(f"{API}/image/{pid}",timeout=3)
            if r.status_code==200: st.image(Image.open(io.BytesIO(r.content)),use_container_width=True)
        except: st.markdown("🖼️")
        bar=int(score*100); rt_tag="🔴 " if is_rt else ""
        st.markdown(
            f"<div style='font-size:11px;color:#e0e0e0;font-weight:600;'>#{rank} {rt_tag}{name}</div>"
            f"<div style='font-size:10px;color:#8B949E;'>{atype} · {clr}</div>"
            f"<div style='margin:2px 0;'><span class='brand-pill'>🏷 {brand}</span></div>"
            f"<div style='background:#1a2235;border-radius:3px;height:4px;'>"
            f"<div style='background:linear-gradient(90deg,{GREEN},{TEAL});width:{bar}%;height:4px;border-radius:3px;'></div></div>"
            f"<div style='font-size:10px;color:{GREEN};font-weight:600;'>Score: {score:.3f}</div>",
            unsafe_allow_html=True)
        if show_exp and expl:
            st.markdown(f"<div class='expl'>💡 {expl}</div>", unsafe_allow_html=True)

def status_bar(label, latency, n, color, mmr_on=False):
    mmr = f"&nbsp;|&nbsp; MMR <span style='color:{GREEN};'>ON</span>" if mmr_on else ""
    st.markdown(f"<div class='status-bar'><span style='color:{color};font-weight:700;'>{label}</span>"
                f"&nbsp; ⏱ <code>{latency} ms</code>&nbsp; 📦 <code>{n} results</code>{mmr}</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERING BASED ON SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

# ════ 🔍 SEARCH ════════════════════════════════════════════════════════════════
if ACTIVE == "🔍 Search":
    st.markdown("<div class='section-title'>🔍 Visual Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload image · Paste URL · Enter Product ID · Fuse multiple images</div>", unsafe_allow_html=True)

    qmode = st.radio("Query Type", ["📁 Upload Image","🌐 Image URL","🔢 Product ID","🔀 Multi-Image Fusion"], horizontal=True, key="s_qmode")

    if qmode == "📁 Upload Image":
        uploaded = st.file_uploader("Drop a fashion image", type=["jpg","jpeg","png"], key="s_upload")
        if uploaded:
            c1,c2 = st.columns([1,4])
            with c1: st.image(Image.open(uploaded), use_container_width=True)
            with c2: st.markdown(f"**{uploaded.name}** · {uploaded.size/1024:.1f} KB · `{method}` · MMR `{'ON' if use_mmr else 'OFF'}`")
    elif qmode == "🌐 Image URL":
        url_in = st.text_input("Paste image URL", placeholder="https://example.com/jacket.jpg", key="s_url")
        if url_in:
            try:
                r=requests.get(url_in,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
                st.image(Image.open(io.BytesIO(r.content)), width=200)
            except: st.info("Preview unavailable — will search anyway.")
    elif qmode == "🔢 Product ID":
        pid_in = st.number_input("Product ID", min_value=1, value=1163, step=1, key="s_pid")
        if st.button("👁 Preview", key="s_prev"):
            try:
                pm=requests.get(f"{API}/product/{int(pid_in)}",timeout=5).json()
                ir=requests.get(f"{API}/image/{int(pid_in)}",timeout=5)
                c1,c2=st.columns([1,4])
                with c1:
                    if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),use_container_width=True)
                with c2: st.markdown(f"**{pm.get('name','')}**\n{pm.get('articleType','')} · {pm.get('baseColour','')} · 🏷 {pm.get('brand','')}")
            except Exception as e: st.error(str(e))
    else:
        st.markdown("Upload **2–4 images** to fuse their styles into one query.")
        multi = st.file_uploader("Upload 2–4 images", type=["jpg","jpeg","png"], accept_multiple_files=True, key="s_multi")
        if multi and len(multi)>=2:
            cols=st.columns(len(multi))
            for i,f in enumerate(multi):
                with cols[i]: st.image(Image.open(f),caption=f"Image {i+1}",use_container_width=True)

    if st.button("🔍 Search Now", type="primary", use_container_width=True, key="s_btn"):
        p={"k":k,"method":method,"use_mmr":use_mmr,"mmr_lambda":mmr_lambda}
        if cat_param:   p["category"]=cat_param
        if color_param: p["color"]=color_param
        if brand_param: p["brand"]=brand_param
        with st.spinner("Searching 44,419 products..."):
            try:
                if qmode=="📁 Upload Image" and 's_upload' in st.session_state and st.session_state.s_upload:
                    up=st.session_state.s_upload; up.seek(0)
                    resp=requests.post(f"{API}/search/upload",files={"file":(up.name,up.getvalue(),"image/jpeg")},params=p,timeout=60).json()
                elif qmode=="🌐 Image URL":
                    url_val=st.session_state.get("s_url","")
                    resp=requests.post(f"{API}/search/url",params={**p,"image_url":url_val},timeout=60).json()
                elif qmode=="🔢 Product ID":
                    resp=requests.post(f"{API}/search/by_id",params={**p,"product_id":int(st.session_state.get("s_pid",1163))},timeout=60).json()
                elif qmode=="🔀 Multi-Image Fusion" and 's_multi' in st.session_state and st.session_state.s_multi and len(st.session_state.s_multi)>=2:
                    fl=[("files",(f.name,f.getvalue(),"image/jpeg")) for f in st.session_state.s_multi]
                    resp=requests.post(f"{API}/search/multi_image",files=fl,params={"k":k,"method":method,"use_mmr":True},timeout=60).json()
                else:
                    st.warning("Provide a query first."); st.stop()

                results=resp.get("results",[]); lat=resp.get("latency_ms",0)
                mlabels={"faiss":"⚡ FAISS (ANN)","graph":"🕸️ Graph (k-NN)","baseline":"🎯 Baseline"}
                mcolors={"faiss":PINK,"graph":GREEN,"baseline":BLUE}
                status_bar(mlabels.get(method,method),lat,len(results),mcolors.get(method,BLUE),use_mmr)
                grid=st.columns(5)
                for i,item in enumerate(results[:10]): render_card(item,grid[i%5])
            except Exception as e: st.error(f"Search failed: {e}")

# ════ 🏷️ BRAND COMPARE ══════════════════════════════════════════════════════════
elif ACTIVE == "🏷️ Brands":
    st.markdown("<div class='section-title'>🏷️ Cross-Brand Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Find the most similar item from each brand side by side. No other platform does this.</div>", unsafe_allow_html=True)
    c1,c2=st.columns([1,2])
    with c1:
        bc_pid=st.number_input("Query Product ID",min_value=1,value=1163,step=1,key="bc_pid")
        bc_k=st.slider("Results per brand",1,5,2,key="bc_k")
    with c2:
        sel_brands=st.multiselect("Brands to compare",options=all_brands,
                                   default=[b for b in ["Nike","Puma","Adidas","Reebok"] if b in all_brands][:4],
                                   max_selections=8,key="bc_brands")
    if st.button("⚡ Compare Brands",type="primary",use_container_width=True,key="bc_btn"):
        if not sel_brands: st.warning("Select at least one brand.")
        else:
            with st.spinner("Comparing across brands..."):
                try:
                    r=requests.get(f"{API}/brand_compare/{int(bc_pid)}",
                                   params={"brands":",".join(sel_brands),"k_per_brand":bc_k,"method":method},
                                   timeout=60).json()
                    qp=r.get("query_product",{})
                    st.markdown(f"**Query:** {qp.get('name','')} · {qp.get('articleType','')} · 🏷 **{qp.get('brand','')}**")
                    st.markdown("---")
                    brand_results=r.get("brand_results",{})
                    bcols=st.columns(len(sel_brands))
                    for i,brand in enumerate(sel_brands):
                        with bcols[i]:
                            br=brand_results.get(brand,{})
                            st.markdown(f"<div style='text-align:center;background:#12192b;border-radius:10px;padding:10px;margin-bottom:10px;border:1px solid #2a3550;'>"
                                        f"<div style='color:{GOLD};font-size:15px;font-weight:800;'>{brand}</div>"
                                        f"<div style='color:#8B949E;font-size:11px;'>{br.get('count',0):,} products · {br.get('latency_ms',0):.1f}ms</div>"
                                        f"</div>",unsafe_allow_html=True)
                            for item in br.get("results",[]):
                                pid=item.get("id"); score=item.get("score",0)
                                try:
                                    ir=requests.get(f"{API}/image/{pid}",timeout=3)
                                    if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),use_container_width=True)
                                except: st.markdown("🖼️")
                                st.markdown(f"<div style='font-size:10px;color:#e0e0e0;'>{item.get('name','')[:24]}</div>"
                                            f"<div style='font-size:10px;color:{GREEN};font-weight:600;'>Score: {score:.3f}</div>"
                                            f"<div class='expl'>💡 {item.get('explanation','')}</div>",unsafe_allow_html=True)
                                st.markdown("---")
                except Exception as e: st.error(f"Brand compare failed: {e}")

# ════ 🎨 STYLE TRANSFER ════════════════════════════════════════════════════════
elif ACTIVE == "🎨 Style":
    st.markdown("<div class='section-title'>🎨 Style Transfer Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Find the same style in a completely different color. Unique to FashionFinder.</div>", unsafe_allow_html=True)
    c1,c2=st.columns([1,1])
    with c1:
        st_pid=st.number_input("Query Product ID",min_value=1,value=1163,step=1,key="st_pid")
        st_mode=st.radio("Mode",["Single target color","All color variants"],key="st_mode")
    with c2:
        color_opts=filt_colors if filt_colors else ["Black","White","Red","Blue","Green","Navy Blue","Grey"]
        st_color=st.selectbox("Target Color",color_opts,key="st_color")
        st_k=st.slider("Results",1,10,5,key="st_k")
    if st.button("🎨 Find Style Transfer",type="primary",use_container_width=True,key="st_btn"):
        with st.spinner("Searching color variants..."):
            try:
                all_var=st_mode=="All color variants"
                p={"product_id":int(st_pid),"k":st_k,"all_variants":all_var}
                if not all_var: p["target_color"]=st_color
                if cat_param: p["category"]=cat_param
                r=requests.get(f"{API}/style_transfer",params=p,timeout=30).json()
                if "error" in r: st.error(r["error"])
                elif all_var:
                    variants=r.get("variants",{})
                    n_colors=len(variants)
                    if n_colors==0: st.warning("No color variants found. Try removing filters.")
                    else:
                        st.success(f"Found same style across **{n_colors} colors**")
                        color_list=list(variants.items())
                        for row_start in range(0,len(color_list),5):
                            row=color_list[row_start:row_start+5]
                            rcols=st.columns(len(row))
                            for j,(cname,items) in enumerate(row):
                                with rcols[j]:
                                    st.markdown(f"<div style='text-align:center;color:{TEAL};font-weight:700;font-size:12px;margin-bottom:4px;'>{cname}</div>",unsafe_allow_html=True)
                                    if items:
                                        item=items[0]; pid=item.get("id")
                                        try:
                                            ir=requests.get(f"{API}/image/{pid}",timeout=3)
                                            if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),use_container_width=True)
                                        except: st.markdown("🖼️")
                                        st.caption(f"{item.get('name','')[:20]}\n{item.get('score',0):.3f}")
                else:
                    results=r.get("results",[]); query=r.get("query_product",{})
                    if not results: st.warning(f"No results in {st_color}. Try a different color.")
                    else:
                        cq,cr=st.columns([1,5])
                        with cq:
                            try:
                                ir=requests.get(f"{API}/image/{query.get('id')}",timeout=3)
                                if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),use_container_width=True)
                            except: pass
                            st.caption(f"Original\n{query.get('baseColour','')}")
                        with cr:
                            st.markdown(f"**Same style in {st_color}:**")
                            rcols=st.columns(5)
                            for i,item in enumerate(results[:5]): render_card(item,rcols[i])
            except Exception as e: st.error(f"Style transfer failed: {e}")

# ════ 👔 OUTFIT FINDER ════════════════════════════════════════════════════════
elif ACTIVE == "👔 Outfit":
    st.markdown("<div class='section-title'>👔 Outfit Completion</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Given any product, FashionFinder suggests complementary items to complete the outfit.</div>", unsafe_allow_html=True)
    of_pid=st.number_input("Product ID",min_value=1,value=1163,step=1,key="of_pid")
    of_k=st.slider("Suggestions per type",1,5,2,key="of_k")
    if st.button("👔 Build Outfit",type="primary",use_container_width=True,key="of_btn"):
        with st.spinner("Building outfit..."):
            try:
                r=requests.get(f"{API}/outfit/{int(of_pid)}",params={"k":of_k},timeout=30).json()
                qp=r.get("query_product",{}); sugg=r.get("suggestions",[]); types=r.get("complementary_types",[])
                cq,ci=st.columns([1,3])
                with cq:
                    try:
                        ir=requests.get(f"{API}/image/{qp.get('id')}",timeout=3)
                        if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),use_container_width=True)
                    except: pass
                with ci:
                    st.markdown(f"**{qp.get('name','')}**")
                    st.markdown(f"Type: {qp.get('articleType','')} · Color: {qp.get('baseColour','')} · 🏷 {qp.get('brand','')}")
                    st.markdown(f"Suggesting: **{', '.join(types)}**")
                st.markdown("---")
                for comp in types:
                    items=[s for s in sugg if s.get("complement_for")==comp]
                    if not items: continue
                    st.markdown(f"#### 👉 {comp}")
                    ocols=st.columns(min(len(items),5))
                    for i,item in enumerate(items[:5]): render_card(item,ocols[i])
            except Exception as e: st.error(f"Outfit failed: {e}")

# ════ 🧬 VISUAL DNA ════════════════════════════════════════════════════════════
elif ACTIVE == "🧬 DNA":
    st.markdown("<div class='section-title'>🧬 Visual DNA</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>GradCAM attention map + 512-dim embedding heatmap. No other platform shows you this.</div>", unsafe_allow_html=True)
    dna_mode=st.radio("Input",["Product ID","Upload Image"],horizontal=True,key="dna_mode")
    dna_pid_val=0; dna_file=None
    if dna_mode=="Product ID":
        dna_pid_val=st.number_input("Product ID",min_value=1,value=1163,step=1,key="dna_pid")
    else:
        dna_file=st.file_uploader("Upload image",type=["jpg","jpeg","png"],key="dna_up")
    if st.button("🧬 Analyze Visual DNA",type="primary",use_container_width=True,key="dna_btn"):
        with st.spinner("Running GradCAM analysis..."):
            try:
                if dna_mode=="Product ID":
                    r=requests.post(f"{API}/visual_dna",params={"product_id":int(dna_pid_val)},timeout=60)
                else:
                    if not dna_file: st.warning("Upload an image first."); st.stop()
                    dna_file.seek(0)
                    r=requests.post(f"{API}/visual_dna",files={"file":(dna_file.name,dna_file.getvalue(),"image/jpeg")},timeout=60)
                if r.status_code!=200: st.error(f"Visual DNA error: {r.status_code}"); st.stop()
                data=r.json()
                dc1,dc2=st.columns(2)
                with dc1:
                    st.markdown("#### 🔴 GradCAM Attention Overlay")
                    st.caption("Red = ResNet18 paid most attention here")
                    if dna_mode=="Product ID":
                        ov=requests.get(f"{API}/visual_dna/overlay",params={"product_id":int(dna_pid_val)},timeout=30)
                    else:
                        dna_file.seek(0)
                        ov=requests.post(f"{API}/visual_dna/overlay_upload",files={"file":(dna_file.name,dna_file.getvalue(),"image/jpeg")},timeout=30)
                    if ov.status_code==200: st.image(Image.open(io.BytesIO(ov.content)),use_container_width=True)
                    st.markdown("**Top attention regions:**")
                    for reg in data.get("top_regions",[]):
                        bar=int(reg.get("strength",0)*100)
                        st.markdown(f"<div style='font-size:11px;color:#e0e0e0;margin-bottom:3px;'>"
                                    f"📍 {reg.get('region','')} — {reg.get('strength',0):.3f}"
                                    f"<div style='background:#1a2235;border-radius:3px;height:4px;'>"
                                    f"<div style='background:{PINK};width:{bar}%;height:4px;border-radius:3px;'></div></div></div>",
                                    unsafe_allow_html=True)
                with dc2:
                    st.markdown("#### 🟢 512-dim Embedding Heatmap")
                    st.caption("Green = strong positive, Red = negative activation")
                    if dna_mode=="Product ID":
                        hm=requests.get(f"{API}/visual_dna/heatmap",params={"product_id":int(dna_pid_val)},timeout=30)
                    else:
                        dna_file.seek(0)
                        hm=requests.post(f"{API}/visual_dna/heatmap_upload",files={"file":(dna_file.name,dna_file.getvalue(),"image/jpeg")},timeout=30)
                    if hm.status_code==200: st.image(Image.open(io.BytesIO(hm.content)),use_container_width=True)
                    stats=data.get("embedding_stats",{})
                    if stats:
                        st.markdown("**Embedding statistics:**")
                        st.dataframe(pd.DataFrame([{
                            "Mean":round(stats.get("mean",0),4),"Std":round(stats.get("std",0),4),
                            "Max":round(stats.get("max",0),4),"Min":round(stats.get("min",0),4),
                            "L2 Norm":round(stats.get("l2_norm",0),4),"Non-zero":stats.get("nonzero_dims",0),
                        }]),use_container_width=True,hide_index=True)
            except Exception as e: st.error(f"Visual DNA error: {e}")

# ════ 🤖 CLIP SEARCH ═══════════════════════════════════════════════════════════
elif ACTIVE == "🤖 CLIP":
    st.markdown("<div class='section-title'>🤖 CLIP Multimodal Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Combine image + text for state-of-the-art visual search. Example: upload a jacket → type 'more formal'.</div>", unsafe_allow_html=True)
    if not health.get("clip_loaded",False):
        st.info("Install CLIP: `pip install git+https://github.com/openai/CLIP.git` then restart the API.")
    clip_img=st.file_uploader("Query image (optional)",type=["jpg","jpeg","png"],key="clip_img")
    clip_txt=st.text_input("Text modifier",placeholder="but make it more formal and elegant",key="clip_txt_v2")
    style_mods={"More Formal":"formal elegant professional","More Casual":"casual relaxed comfortable",
                "More Sporty":"athletic sports performance","More Elegant":"luxury sophisticated premium",
                "More Colorful":"vibrant colorful bright bold","More Minimal":"minimal simple clean",
                "Party Style":"party evening glamorous festive","Vintage Style":"vintage retro classic heritage"}
    st.markdown("**Quick modifiers:**")
    mc=st.columns(4)
    for i,(name,desc) in enumerate(style_mods.items()):
        with mc[i%4]:
            if st.button(name,key=f"cm_{i}",use_container_width=True):
                st.rerun()
    img_w=st.slider("Image weight",0.0,1.0,0.7,0.05,key="clip_imgw")
    st.caption(f"Image: {img_w:.2f} | Text: {1-img_w:.2f}")
    if st.button("🤖 CLIP Search",type="primary",use_container_width=True,
                  disabled=not health.get("clip_loaded",False),key="clip_btn"):
        with st.spinner("CLIP multimodal search..."):
            try:
                p={"k":k,"method":method,"image_weight":img_w,"text_weight":round(1-img_w,2)}
                if cat_param: p["category"]=cat_param
                clip_txt_val=st.session_state.get("clip_txt","")
                if clip_img and clip_txt_val:
                    clip_img.seek(0)
                    r=requests.post(f"{API}/clip_search",files={"file":(clip_img.name,clip_img.getvalue(),"image/jpeg")},params={**p,"text":clip_txt_val},timeout=60).json()
                elif clip_img:
                    clip_img.seek(0)
                    r=requests.post(f"{API}/clip_search",files={"file":(clip_img.name,clip_img.getvalue(),"image/jpeg")},params=p,timeout=60).json()
                elif clip_txt_val:
                    r=requests.post(f"{API}/clip_search",params={**p,"text":clip_txt_val},timeout=60).json()
                else:
                    st.warning("Provide image or text."); st.stop()
                results=r.get("results",[]); latency=r.get("latency_ms",0)
                status_bar(f"🤖 CLIP ({r.get('mode','')})",latency,len(results),PURPLE)
                grid=st.columns(5)
                for i,item in enumerate(results[:10]): render_card(item,grid[i%5])
            except Exception as e: st.error(f"CLIP failed: {e}")

# ════ ➕ ADD PRODUCT ══════════════════════════════════════════════════════════
elif ACTIVE == "➕ Add":
    st.markdown("<div class='section-title'>➕ Real-Time Product Index</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Add any image — immediately searchable, no index rebuild needed.</div>", unsafe_allow_html=True)
    c1,c2=st.columns([1,1])
    with c1:
        new_img=st.file_uploader("Product Image",type=["jpg","jpeg","png"],key="add_img")
        if new_img: st.image(Image.open(new_img),caption="Preview",use_container_width=True)
    with c2:
        new_name=st.text_input("Product Name",value="My Custom Product",key="add_name")
        new_cat=st.selectbox("Category",categories if categories else ["Apparel"],key="add_cat")
        new_color=st.selectbox("Color",filt_colors if filt_colors else ["Black","White","Blue","Red"],key="add_color")
        new_brand=st.text_input("Brand",value="Custom",key="add_brand")
        new_type=st.text_input("Article Type",value="Tshirts",key="add_type")
    if st.button("➕ Add to Index",type="primary",use_container_width=True,key="add_btn"):
        if not new_img: st.warning("Upload a product image first.")
        elif not new_name.strip(): st.warning("Enter a product name.")
        else:
            with st.spinner("Embedding and indexing..."):
                try:
                    new_img.seek(0)
                    r=requests.post(f"{API}/realtime/add",
                                     files={"file":(new_img.name,new_img.getvalue(),"image/jpeg")},
                                     params={"name":new_name,"category":new_cat,"color":new_color,"brand":new_brand,"article_type":new_type},
                                     timeout=30).json()
                    st.success(f"✅ Added! Product ID: **{r.get('id','?')}** — Immediately searchable!")
                    st.json(r)
                except Exception as e: st.error(f"Add failed: {e}")
    st.markdown("---")
    st.markdown("#### 🔴 Live Buffer")
    try:
        rt_data=requests.get(f"{API}/realtime/products",timeout=5).json()
        rt_prods=rt_data.get("products",[])
        if rt_prods:
            st.markdown(f"**{len(rt_prods)} products** immediately searchable:")
            rtcols=st.columns(min(len(rt_prods),5))
            for i,p in enumerate(rt_prods[:10]):
                with rtcols[i%5]:
                    pid=p.get("id")
                    try:
                        ir=requests.get(f"{API}/image/{pid}",timeout=2)
                        if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),use_container_width=True)
                    except: st.markdown("🖼️")
                    st.caption(f"🔴 {p.get('productDisplayName','')[:22]}")
            if st.button("🗑️ Clear Buffer",key="clear_buf"):
                requests.delete(f"{API}/realtime/clear",timeout=5)
                st.success("Buffer cleared."); st.rerun()
        else: st.info("No products in buffer yet.")
    except: st.info("Real-time index not available.")

# ════ 📊 BENCHMARK ════════════════════════════════════════════════════════════
elif ACTIVE == "📊 Benchmark":
    st.markdown("<div class='section-title'>📊 Live Benchmark</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>All 3 methods benchmarked live. Every number is real and reproducible.</div>", unsafe_allow_html=True)
    bench_n=st.number_input("Number of queries",10,500,50,step=10,key="bench_n")
    if st.button("▶️ Run Benchmark",type="primary",key="bench_btn"):
        with st.spinner(f"Running {bench_n} queries..."):
            try:
                bd=requests.post(f"{API}/benchmark",params={"n":bench_n,"k":k},timeout=300).json()
                st.session_state["bench_data"]=bd
            except Exception as e: st.error(f"Benchmark failed: {e}")
    if "bench_data" in st.session_state:
        bd=st.session_state["bench_data"]
        mc=st.columns(6)
        for col,label,val,color in [
            (mc[0],"Baseline Median",f"{bd['baseline_median_ms']} ms",BLUE),
            (mc[1],"Graph Median",f"{bd['graph_median_ms']} ms",GREEN),
            (mc[2],"FAISS Median",f"{bd['faiss_median_ms']} ms",PINK),
            (mc[3],"Baseline p95",f"{bd['baseline_p95_ms']} ms",BLUE),
            (mc[4],"Graph p95",f"{bd['graph_p95_ms']} ms",GREEN),
            (mc[5],"FAISS p95",f"{bd['faiss_p95_ms']} ms",PINK),
        ]:
            with col: st.markdown(f"<div class='metric-box'><div style='color:#aaa;font-size:10px;'>{label}</div><div style='color:{color};font-size:20px;font-weight:800;'>{val}</div></div>",unsafe_allow_html=True)
        st.markdown("---")
        bc1,bc2=st.columns(2)
        with bc1:
            st.markdown("#### Latency (ms)")
            fig,ax=plt.subplots(figsize=(5,3.5)); fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#12192b")
            methods=["Baseline","Graph","FAISS"]
            medians=[bd["baseline_median_ms"],bd["graph_median_ms"],bd["faiss_median_ms"]]
            p95s=[bd["baseline_p95_ms"],bd["graph_p95_ms"],bd["faiss_p95_ms"]]
            x=np.arange(3); w=0.35
            ax.bar(x-w/2,medians,w,color=["#4FC3F7","#81C784","#F48FB1"],alpha=0.9,label="Median")
            ax.bar(x+w/2,p95s,w,color=["#4FC3F7","#81C784","#F48FB1"],alpha=0.45,label="p95")
            ax.set_xticks(x); ax.set_xticklabels(methods,color="white")
            ax.set_ylabel("ms",color="white"); ax.tick_params(colors="white")
            ax.legend(facecolor="#12192b",labelcolor="white"); ax.spines[:].set_color("#2a3550")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with bc2:
            st.markdown("#### Recall@k")
            fig2,ax2=plt.subplots(figsize=(5,3.5)); fig2.patch.set_facecolor("#0a0e1a"); ax2.set_facecolor("#12192b")
            recalls=[1.000,bd["graph_recall"],bd["faiss_recall"]]
            bars=ax2.bar(methods,recalls,color=["#4FC3F7","#81C784","#F48FB1"],alpha=0.9,width=0.5)
            ax2.set_ylim(0,1.1); ax2.set_ylabel("Recall@k",color="white"); ax2.tick_params(colors="white")
            ax2.spines[:].set_color("#2a3550"); ax2.set_xticklabels(methods,color="white")
            ax2.axhline(y=1.0,color="#555",linestyle="--",linewidth=1)
            for bar,val in zip(bars,recalls):
                ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f"{val:.3f}",ha="center",va="bottom",color="white",fontsize=11,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.dataframe(pd.DataFrame({"Method":["Baseline","Graph","FAISS"],
            "Median (ms)":[bd["baseline_median_ms"],bd["graph_median_ms"],bd["faiss_median_ms"]],
            "p95 (ms)":[bd["baseline_p95_ms"],bd["graph_p95_ms"],bd["faiss_p95_ms"]],
            "Recall@k":[1.000,bd["graph_recall"],bd["faiss_recall"]],
            "Type":["Exact","Approximate","Approximate"]}),use_container_width=True,hide_index=True)

# ════ 🕸️ GRAPH EXPLORER ══════════════════════════════════════════════════════
elif ACTIVE == "🕸️ Graph":
    st.markdown("<div class='section-title'>🕸️ k-NN Similarity Graph</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Visualize product relationships. No other platform shows you this.</div>", unsafe_allow_html=True)
    gc1,_=st.columns([1,2])
    with gc1:
        ge_pid=st.number_input("Center Product ID",min_value=1,value=1163,step=1,key="ge_pid")
        ge_hops=st.slider("Graph hops",1,3,2,key="ge_hops")
    if st.button("🕸️ Show Graph",type="primary",key="ge_btn"):
        with st.spinner("Building graph..."):
            try:
                gdata=requests.get(f"{API}/graph_neighbors/{int(ge_pid)}",params={"hops":ge_hops},timeout=15).json()
                G=nx.Graph(); cn=gdata["center"]["id"]; G.add_node(cn)
                cmap={"Apparel":"#81C784","Footwear":"#FFB74D","Accessories":"#F48FB1",
                      "Personal Care":"#CE93D8","Sporting Goods":"#80DEEA","Free Items":"#FFF176","Home":"#FFCC80"}
                node_labels={cn:f"#{cn}\nQUERY"}
                for nd in gdata.get("nodes",[]):
                    nid=nd["id"]; G.add_node(nid); G.add_edge(nd["parent"],nid,weight=nd.get("score",0))
                    node_labels[nid]=f"{nd.get('articleType','')[:8]}\n{nd.get('score',0):.2f}"
                n_colors=[]; n_sizes=[]
                for n in G.nodes():
                    if n==cn: n_colors.append("#FFD700"); n_sizes.append(1000)
                    else:
                        cat=next((nd.get("masterCategory","Unknown") for nd in gdata.get("nodes",[]) if nd["id"]==n),"Unknown")
                        n_colors.append(cmap.get(cat,"#90A4AE")); n_sizes.append(350)
                fig,ax=plt.subplots(figsize=(12,7)); fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0a0e1a")
                pos=nx.spring_layout(G,seed=42,k=1.5)
                nx.draw_networkx_nodes(G,pos,node_color=n_colors,node_size=n_sizes,ax=ax,alpha=0.92)
                nx.draw_networkx_edges(G,pos,edge_color="#2a3550",ax=ax,alpha=0.6,width=1.2)
                nx.draw_networkx_labels(G,pos,labels=node_labels,font_size=6,font_color="white",ax=ax)
                patches=[mpatches.Patch(color=c,label=cat) for cat,c in cmap.items()]
                patches.append(mpatches.Patch(color="#FFD700",label="Query (center)"))
                ax.legend(handles=patches,loc="upper left",facecolor="#12192b",labelcolor="white",fontsize=8,framealpha=0.8)
                ax.axis("off"); ax.set_title(f"k-NN Graph: Product {ge_pid} | {ge_hops} hops | {G.number_of_nodes()} nodes",color=BLUE,fontsize=13,fontweight="bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()
                rows=[{"ID":nd["id"],"Name":nd.get("name","")[:40],"Brand":nd.get("brand",""),
                       "Category":nd.get("masterCategory",""),"Score":nd.get("score",0),"Depth":nd.get("depth",1)}
                      for nd in gdata.get("nodes",[])]
                if rows: st.dataframe(pd.DataFrame(rows).sort_values("Score",ascending=False),use_container_width=True,hide_index=True)
            except Exception as e: st.error(f"Graph failed: {e}")

# ════ 📅 FASHION TIMELINE ══════════════════════════════════════════════════════
elif ACTIVE == "📅 Timeline":
    st.markdown("<div class='section-title'>📅 Fashion Timeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>How colors, styles, and volumes evolved year by year in the dataset.</div>", unsafe_allow_html=True)
    if st.button("📅 Load Fashion Timeline",type="primary",key="tl_btn"):
        with st.spinner("Analyzing trends..."):
            try:
                tl=requests.get(f"{API}/timeline",timeout=30).json()
                years=tl.get("years",[])
                if not years: st.warning("No year data available."); st.stop()
                st.success(f"Data for **{len(years)} years**: {years[0]} → {years[-1]}")
                st.markdown("#### 📈 Product Volume by Year")
                vr=requests.get(f"{API}/timeline/volume_chart",timeout=20)
                if vr.status_code==200: st.image(Image.open(io.BytesIO(vr.content)),use_container_width=True)
                tl1,tl2=st.columns(2)
                with tl1:
                    st.markdown("#### 🎨 Color Trends")
                    cr=requests.get(f"{API}/timeline/color_chart",timeout=20)
                    if cr.status_code==200: st.image(Image.open(io.BytesIO(cr.content)),use_container_width=True)
                with tl2:
                    st.markdown("#### 👕 Article Type Trends")
                    ar=requests.get(f"{API}/timeline/article_chart",timeout=20)
                    if ar.status_code==200: st.image(Image.open(io.BytesIO(ar.content)),use_container_width=True)
                st.markdown("---")
                sel_yr=st.select_slider("Explore year",options=years,value=years[-1],key="tl_year")
                yr=requests.get(f"{API}/timeline/year/{sel_yr}",timeout=10).json()
                if "error" not in yr:
                    yc1,yc2,yc3=st.columns(3)
                    with yc1:
                        st.markdown(f"**{sel_yr} — Top Colors**")
                        for item in yr.get("top_colors",[]): st.markdown(f"🎨 **{item['color']}**: {item['count']:,}")
                    with yc2:
                        st.markdown(f"**{sel_yr} — Top Types**")
                        for item in yr.get("top_articles",[]): st.markdown(f"👕 **{item['type']}**: {item['count']:,}")
                    with yc3:
                        st.markdown(f"**{sel_yr} — Top Categories**")
                        for item in yr.get("top_categories",[]): st.markdown(f"📁 **{item['category']}**: {item['count']:,}")
                    st.info(f"Total products in {sel_yr}: **{yr.get('total_products',0):,}**")
            except Exception as e: st.error(f"Timeline failed: {e}")

# ════ 🔥 TRENDING ════════════════════════════════════════════════════════════
elif ACTIVE == "🔥 Trending":
    st.markdown("<div class='section-title'>🔥 Trending Now</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Live search trends — every search you make is tracked here.</div>", unsafe_allow_html=True)
    hours=st.slider("Time window (hours)",1,168,24,key="tr_hours")
    try:
        td=requests.get(f"{API}/trending",params={"hours":hours,"top_n":20},timeout=10).json()
        st.markdown(f"**{td.get('total_searches',0)} total searches** in the last {hours} hours")
        st.markdown("---")
        tc1,tc2,tc3=st.columns(3)
        with tc1:
            st.markdown("#### 🔥 Trending Products")
            prods=td.get("trending_products",[])
            if prods:
                for p in prods[:8]:
                    pid=p.get("id"); views=p.get("views",0)
                    ct,ci=st.columns([3,1])
                    with ct: st.markdown(f"**{p.get('name','')[:28]}**\n🏷 {p.get('brand','')} | 👁 {views}")
                    with ci:
                        try:
                            ir=requests.get(f"{API}/image/{pid}",timeout=2)
                            if ir.status_code==200: st.image(Image.open(io.BytesIO(ir.content)),width=60)
                        except: pass
                    st.markdown("---")
            else: st.info("Do some searches in Search tab to see trends!")
        with tc2:
            st.markdown("#### 📂 Trending Categories")
            cats=td.get("trending_categories",[])
            if cats:
                fig,ax=plt.subplots(figsize=(4,3)); fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#12192b")
                ax.barh([c["category"] for c in cats[:7]],[c["searches"] for c in cats[:7]],color="#4FC3F7",alpha=0.85)
                ax.tick_params(colors="white"); ax.spines[:].set_color("#2a3550"); ax.set_xlabel("Searches",color="white")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else: st.info("No category data yet.")
        with tc3:
            st.markdown("#### 🏷️ Trending Brands")
            bts=td.get("trending_brands",[])
            if bts:
                fig,ax=plt.subplots(figsize=(4,3)); fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#12192b")
                ax.barh([b["brand"] for b in bts[:7]],[b["searches"] for b in bts[:7]],color="#FFD700",alpha=0.85)
                ax.tick_params(colors="white"); ax.spines[:].set_color("#2a3550"); ax.set_xlabel("Searches",color="white")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else: st.info("No brand data yet.")
        st.markdown("---")
        hist=requests.get(f"{API}/search_history",params={"limit":20},timeout=5).json().get("history",[])
        if hist:
            st.markdown("#### 📋 Recent Searches")
            st.dataframe(pd.DataFrame(hist),use_container_width=True,hide_index=True)
    except Exception as e: st.error(f"Trending failed: {e}")

# ════ 💬 AI FASHION ASSISTANT ═══════════════════════════════════════════════
elif ACTIVE == "💬 Assistant":
    st.markdown("<div class='section-title'>💬 AI Fashion Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Chat to find products, build outfits, get style advice for any occasion. Powered by Claude AI.</div>", unsafe_allow_html=True)

    # Init chatbot in session state
    if "chatbot_instance" not in st.session_state:
        try:
            from chatbot import RAGFashionChatbot as FashionChatbot
            from heap_ranker import top_k_cosine as tkc
            import numpy as np
            import pandas as pd

            emb  = np.load("embeddings/embeddings.npy")
            meta = pd.read_csv("embeddings/metadata.csv").reset_index(drop=True)
            from embedder    import Embedder
            from hash_index  import HashIndex
            from color_index import ColorIndex
            from brand_index import BrandIndex
            from faiss_index import FAISSIndex

            hi = HashIndex(meta)
            ci = ColorIndex(meta)
            bi = BrandIndex(meta)
            fi = FAISSIndex()
            if os.path.exists("embeddings/faiss.index"):
                fi.load("embeddings/faiss.index")

            st.session_state["chatbot_instance"] = FashionChatbot(
                metadata=meta, embeddings=emb,
                hash_index=hi, color_index=ci, brand_index=bi,
                heap_ranker_fn=tkc, embedder=Embedder(),
                faiss_index=fi,
                api_key=os.environ.get("ANTHROPIC_API_KEY","")
            )
        except Exception as e:
            st.error(f"Could not load chatbot: {e}")
            st.stop()

    bot = st.session_state["chatbot_instance"]

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_products" not in st.session_state:
        st.session_state["chat_products"] = []

    # Quick question buttons
    st.markdown("**Quick questions — click to ask:**")
    qq_cols = st.columns(4)
    quick_questions = [
        ("👰 Wedding", "What should I wear to a wedding? I prefer traditional Indian outfits"),
        ("💪 Gym", "Build me a complete gym workout outfit"),
        ("💼 Office", "What is a good professional office look?"),
        ("🎉 Party", "Suggest a party outfit for a night out"),
        ("🏖️ Beach", "What should I wear to the beach?"),
        ("❄️ Winter", "Build me a cozy winter outfit"),
        ("🎓 College", "What is a good college outfit for everyday wear?"),
        ("💕 Date Night", "Help me pick a date night outfit"),
    ]
    for i, (label, question) in enumerate(quick_questions):
        with qq_cols[i % 4]:
            if st.button(label, key=f"qq_{i}", use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()

    # Process pending question from quick buttons
    if "pending_question" in st.session_state:
        q = st.session_state.pop("pending_question")
        with st.spinner("Thinking..."):
            response, products, tts_js = bot.chat(q)
            st.session_state["chat_history"].append(("user", q))
            st.session_state["chat_history"].append(("assistant", response))
            st.session_state["chat_products"] = products
            st.session_state["chat_tts"] = tts_js
        st.rerun()

    st.markdown("---")

    # Chat layout
    chat_col, prod_col = st.columns([3, 2])

    with chat_col:
        # Display conversation
        st.markdown("#### Conversation")
        chat_container = st.container()
        with chat_container:
            for role, msg in st.session_state["chat_history"]:
                if role == "user":
                    st.markdown(
                        f"<div style='background:#1a2235;border-radius:10px 10px 2px 10px;"
                        f"padding:10px 14px;margin:6px 0;margin-left:20%;border:1px solid #2a3550;'>"
                        f"<div style='color:#8B949E;font-size:10px;margin-bottom:4px;'>You</div>"
                        f"<div style='color:#e0e0e0;font-size:13px;'>{msg}</div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,#0d2137,#0d3730);"
                        f"border-radius:10px 10px 10px 2px;padding:10px 14px;margin:6px 0;"
                        f"margin-right:20%;border:1px solid #1e4d3b;'>"
                        f"<div style='color:{TEAL};font-size:10px;margin-bottom:4px;'>👗 FashionFinder AI</div>"
                        f"<div style='color:#e0e0e0;font-size:13px;'>{msg}</div></div>",
                        unsafe_allow_html=True
                    )

        # Input
        st.markdown("")
        user_input = st.text_input(
            "Type your message",
            placeholder="e.g. Show me red Nike shoes, What to wear for a date night...",
            key="chat_input_box",
            label_visibility="collapsed"
        )

        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            send_btn = st.button("💬 Send", type="primary", use_container_width=True, key="chat_send")
        with c2:
            reset_btn = st.button("🔄 Reset", use_container_width=True, key="chat_reset")
        with c3:
            tts_enabled = st.toggle("🔊 Speak", value=True, key="tts_on")

        if send_btn and user_input.strip():
            with st.spinner("Thinking..."):
                response, products, tts_js = bot.chat(user_input)
                st.session_state["chat_history"].append(("user", user_input))
                st.session_state["chat_history"].append(("assistant", response))
                st.session_state["chat_products"] = products
                st.session_state["chat_tts"] = tts_js
            st.rerun()

        # TTS: speak last response
        if st.session_state.get("chat_tts") and tts_enabled:
            import streamlit.components.v1 as components
            components.html(st.session_state.pop("chat_tts",""), height=0)

        if reset_btn:
            bot.reset()
            st.session_state["chat_history"] = []
            st.session_state["chat_products"] = []
            st.rerun()

    with prod_col:
        st.markdown("#### Products from Chat")
        chat_prods = st.session_state.get("chat_products", [])
        if chat_prods:
            st.markdown(f"**{len(chat_prods)} products found:**")
            for p in chat_prods:
                pid = p.get("id")
                pc1, pc2 = st.columns([1, 2])
                with pc1:
                    try:
                        ir = requests.get(f"{API}/image/{pid}", timeout=2)
                        if ir.status_code == 200:
                            st.image(Image.open(io.BytesIO(ir.content)), use_container_width=True)
                    except: st.markdown("🖼️")
                with pc2:
                    st.markdown(
                        f"<div style='font-size:11px;color:#e0e0e0;font-weight:600;'>{p.get('name','')[:30]}</div>"
                        f"<div style='font-size:10px;color:#8B949E;'>{p.get('type','')} · {p.get('color','')}</div>"
                        f"<div style='font-size:10px;color:{GOLD};'>🏷 {p.get('brand','')}</div>"
                        f"<div style='font-size:10px;color:{GREEN};'>ID: {pid}</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("---")
        else:
            st.markdown(
                f"<div style='background:#12192b;border:1px dashed #2a3550;border-radius:10px;"
                f"padding:20px;text-align:center;color:#555;'>"
                f"Products mentioned in chat will appear here</div>",
                unsafe_allow_html=True
            )
        
        # Suggested follow-ups
        if st.session_state["chat_history"]:
            st.markdown("---")
            st.markdown("**Follow-up suggestions:**")
            suggestions = [
                "Show me similar items in a different color",
                "What shoes go with this?",
                "Find me something more formal",
                "Show me options from Nike",
                "What accessories match this outfit?",
            ]
            for sug in suggestions[:3]:
                if st.button(f"💡 {sug}", key=f"sug_{hash(sug)}", use_container_width=True):
                    st.session_state["pending_question"] = sug
                    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,#2a3550,transparent);margin:20px 0 8px;'>
<div style='text-align:center;color:#333;font-size:11px;'>
FashionFinder v4.0 Ultimate · Built by
<b style='color:{BLUE};'>Akila Lourdes Miriyala Francis</b> &
<b style='color:{GREEN};'>Akilan Manivannan</b> · LIU Brooklyn · 2026
</div>
""", unsafe_allow_html=True)
