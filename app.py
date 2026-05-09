"""
app.py - FashionFinder v4.0 Ultimate
Hugging Face Spaces deployment
Single file: FastAPI backend + Gradio UI combined
"""

import os
import sys
import io
import time
import random
import threading
import numpy as np
import pandas as pd
from PIL import Image
import gradio as gr
import requests as req_lib

# ── Load all components ────────────────────────────────────────────────────────
EMBEDDINGS_PATH = "embeddings/embeddings.npy"
METADATA_PATH   = "embeddings/metadata.csv"
GRAPH_PATH      = "embeddings/knn_graph.pkl"
FAISS_PATH      = "embeddings/faiss.index"
IMAGES_DIR      = "archive-2/images"

print("Loading FashionFinder v4.0...")

embeddings = np.load(EMBEDDINGS_PATH)
metadata   = pd.read_csv(METADATA_PATH).reset_index(drop=True)

from embedder           import Embedder
from hash_index         import HashIndex
from color_index        import ColorIndex
from brand_index        import BrandIndex
from knn_graph          import KNNGraph
from faiss_index        import FAISSIndex
from heap_ranker        import top_k_cosine
from explainer          import SimilarityExplainer
from mmr_reranker       import apply_mmr
from style_transfer_search import StyleTransferSearch

embedder     = Embedder()
hash_index   = HashIndex(metadata)
color_index  = ColorIndex(metadata)
brand_index  = BrandIndex(metadata)
explainer    = SimilarityExplainer(metadata, embeddings)
style_search = StyleTransferSearch(embeddings, metadata, color_index, brand_index)

knn_graph = KNNGraph()
if os.path.exists(GRAPH_PATH):
    knn_graph.load(GRAPH_PATH)

faiss_index = FAISSIndex()
if os.path.exists(FAISS_PATH):
    faiss_index.load(FAISS_PATH)

print(f"Ready: {len(embeddings):,} products | {len(brand_index.all_brands())} brands")

# Init chatbot
from chatbot import FashionChatbot
chatbot = FashionChatbot(
    metadata=metadata,
    embeddings=embeddings,
    hash_index=hash_index,
    color_index=color_index,
    brand_index=brand_index,
    heap_ranker_fn=top_k_cosine,
    embedder=embedder,
    faiss_index=faiss_index,
    api_key=os.environ.get("ANTHROPIC_API_KEY","")
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def idx_from_id(product_id):
    m = metadata[metadata["id"] == int(product_id)].index.tolist()
    return m[0] if m else None

def to_meta(idx):
    row = metadata.loc[idx]
    pid = int(row.get("id", idx))
    return {
        "id": pid, "idx": int(idx),
        "name":     str(row.get("productDisplayName","Unknown")),
        "category": str(row.get("masterCategory","Unknown")),
        "type":     str(row.get("articleType","Unknown")),
        "color":    str(row.get("baseColour","Unknown")),
        "brand":    brand_index.get_brand(idx),
        "img_path": f"{IMAGES_DIR}/{pid}.jpg",
    }

def get_image(pid):
    p = f"{IMAGES_DIR}/{pid}.jpg"
    if os.path.exists(p):
        return Image.open(p)
    return None

def do_search(query_vec, k, method, category=None, color=None, brand=None,
              query_idx=None, use_mmr=False, mmr_lambda=0.6):
    # Filter candidates
    if brand:
        cands = list(set(brand_index.get_indices(brand=brand, category=category)))
    elif category or color:
        cands = color_index.get_indices(category=category, color=color)
    else:
        cands = None

    t0 = time.perf_counter()
    if method == "faiss" and faiss_index and faiss_index.index:
        top_k = faiss_index.search(query_vec, cands, k*2 if use_mmr else k)
        top_k = [(i,s) for i,s in top_k if i != query_idx]
    elif method == "graph" and knn_graph:
        entry = cands if cands else list(range(min(500, len(embeddings))))
        if query_idx: entry = [i for i in entry if i != query_idx]
        top_k = knn_graph.search(query_vec, embeddings, entry, k*2 if use_mmr else k)
        top_k = [(i,s) for i,s in top_k if i != query_idx]
    else:
        c = cands if cands else list(range(len(embeddings)))
        if query_idx: c = [i for i in c if i != query_idx]
        top_k = top_k_cosine(query_vec, embeddings, c, k*2 if use_mmr else k)

    if use_mmr and top_k:
        top_k = apply_mmr(query_vec, top_k, embeddings, k=k, lambda_param=mmr_lambda)

    latency = (time.perf_counter() - t0) * 1000
    return top_k[:k], round(latency, 2)

def results_to_gallery(top_k):
    """Convert results to list of (image, caption) for Gradio gallery."""
    gallery = []
    for rank, (idx, score) in enumerate(top_k, 1):
        m = to_meta(idx)
        img = get_image(m["id"])
        if img:
            caption = f"#{rank} {m['name'][:30]}\n{m['type']} | {m['color']}\n🏷 {m['brand']} | Score: {score:.3f}"
            gallery.append((img, caption))
    return gallery

# ── STYLE MAP for CLIP-style text filtering ────────────────────────────────────
STYLE_MAP = {
    "formal":   ["Shirts","Formal Shoes","Trousers","Blazers","Ties","Belts"],
    "casual":   ["Tshirts","Jeans","Shorts","Casual Shoes","Tops","Flats"],
    "sporty":   ["Sports Shoes","Track Pants","Jackets","Sports Sandals","Socks"],
    "elegant":  ["Sarees","Kurtas","Gowns","Lehenga Cholis","Heels","Dupattas"],
    "party":    ["Dresses","Heels","Tops","Tunics","Skirts","Clutches","Earrings"],
    "minimal":  ["Tshirts","Shirts","Trousers","Jeans","Flats","Sandals"],
    "colorful": ["Tops","Shirts","Tshirts","Dresses","Kurtas","Skirts"],
    "vintage":  ["Jeans","Shirts","Jackets","Casual Shoes","Caps","Sunglasses"],
    "winter":   ["Jackets","Sweaters","Sweatshirts","Boots","Mufflers","Gloves"],
}
SYNONYMS = {
    "professional":"formal","office":"formal","business":"formal",
    "relaxed":"casual","comfortable":"casual","everyday":"casual",
    "athletic":"sporty","gym":"sporty","sport":"sporty","running":"sporty",
    "luxury":"elegant","sophisticated":"elegant","traditional":"elegant",
    "ethnic":"elegant","saree":"elegant","kurti":"elegant",
    "night":"party","evening":"party","festive":"party","glamorous":"party",
    "simple":"minimal","plain":"minimal","basic":"minimal","clean":"minimal",
    "bright":"colorful","vibrant":"colorful","bold":"colorful",
    "retro":"vintage","classic":"vintage","throwback":"vintage",
    "cold":"winter","warm":"winter","cozy":"winter",
}

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def search_by_image(image, method, k, use_mmr, category, color, brand):
    if image is None:
        return [], "Please upload an image."
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    vec = embedder.embed_pil(pil)
    cat = category if category != "All" else None
    clr = color    if color    != "All" else None
    brd = brand    if brand    != "All" else None
    top_k, lat = do_search(vec, k, method, cat, clr, brd, use_mmr=use_mmr)
    gallery = results_to_gallery(top_k)
    info = f"✅ {method.upper()} | {lat}ms | {len(top_k)} results"
    return gallery, info

def search_by_id(product_id, method, k, use_mmr, category, color, brand):
    idx = idx_from_id(product_id)
    if idx is None:
        return [], f"Product ID {product_id} not found."
    vec = embeddings[idx]
    cat = category if category != "All" else None
    clr = color    if color    != "All" else None
    brd = brand    if brand    != "All" else None
    if not cat:
        cat = hash_index.infer_category(metadata, idx)
    top_k, lat = do_search(vec, k, method, cat, clr, brd, query_idx=idx, use_mmr=use_mmr)
    gallery = results_to_gallery(top_k)
    m = to_meta(idx)
    info = f"✅ Query: {m['name'][:40]} | {method.upper()} | {lat}ms | {len(top_k)} results"
    return gallery, info

def search_by_url(url, method, k, use_mmr):
    try:
        r = req_lib.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        pil = Image.open(io.BytesIO(r.content))
        vec = embedder.embed_pil(pil)
        top_k, lat = do_search(vec, k, method)
        gallery = results_to_gallery(top_k)
        return pil, gallery, f"✅ URL Search | {method.upper()} | {lat}ms"
    except Exception as e:
        return None, [], f"❌ Error: {e}"

def search_style_transfer(product_id, target_color, k):
    idx = idx_from_id(product_id)
    if idx is None:
        return [], f"Product ID {product_id} not found."
    vec = embeddings[idx]
    results = style_search.search(vec, target_color, k=k, query_idx=idx)
    gallery = []
    for r in results:
        img = get_image(r["id"])
        if img:
            caption = f"#{r['rank']} {r['name'][:28]}\n{r['articleType']} | {r['baseColour']}\n🏷 {r['brand']} | {r['score']:.3f}"
            gallery.append((img, caption))
    return gallery, f"✅ Style Transfer → {target_color} | {len(results)} results"

def search_clip_style(image, style_modifier, custom_text, k):
    if image is None:
        return [], "Please upload an image."
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    vec = embedder.embed_pil(pil)

    text = custom_text.strip() if custom_text.strip() else style_modifier
    text_lower = text.lower()

    matched_style = None
    for style in STYLE_MAP:
        if style in text_lower:
            matched_style = style; break
    if not matched_style:
        for syn, style in SYNONYMS.items():
            if syn in text_lower:
                matched_style = style; break

    matched_articles = STYLE_MAP.get(matched_style, [])

    if matched_articles:
        article_lower = set(a.lower() for a in matched_articles)
        cands = [i for i in range(len(embeddings))
                 if str(metadata.loc[i].get("articleType","")).lower() in article_lower]
        cands = cands if len(cands) >= k else list(range(len(embeddings)))
    else:
        cands = list(range(len(embeddings)))

    t0 = time.perf_counter()
    top_k = top_k_cosine(vec, embeddings, cands, k)
    lat = round((time.perf_counter()-t0)*1000, 2)

    gallery = results_to_gallery(top_k)
    mode = f"image+style ({matched_style})" if matched_style else "image"
    return gallery, f"✅ CLIP-Style | {mode} | {lat}ms | {len(top_k)} results"

def run_benchmark(n_queries, k):
    n = min(int(n_queries), len(embeddings))
    qidxs = random.sample(range(len(embeddings)), n)
    b_lats, f_lats, g_lats = [], [], []
    f_rec, g_rec = [], []
    for qi in qidxs:
        vec = embeddings[qi]
        cat = hash_index.infer_category(metadata, qi)
        rb, bl = [], 0
        try:
            rb_r, bl = do_search(vec, k, "baseline", cat, query_idx=qi)
            b_lats.append(bl)
            rb = rb_r
        except: pass
        b_ids = set(i for i,_ in rb)
        if faiss_index and faiss_index.index:
            try:
                rf, fl = do_search(vec, k, "faiss", cat, query_idx=qi)
                f_lats.append(fl)
                f_ids = set(i for i,_ in rf)
                f_rec.append(len(b_ids&f_ids)/len(b_ids) if b_ids else 0)
            except: pass
        if knn_graph:
            try:
                rg, gl = do_search(vec, k, "graph", cat, query_idx=qi)
                g_lats.append(gl)
                g_ids = set(i for i,_ in rg)
                g_rec.append(len(b_ids&g_ids)/len(b_ids) if b_ids else 0)
            except: pass

    def p(a, pct): return round(float(np.percentile(a, pct)), 2) if a else 0.0
    result = f"""
## 📊 Benchmark Results ({n} queries, k={k})

| Method | Median (ms) | p95 (ms) | Recall@k |
|--------|-------------|----------|----------|
| Baseline | {p(b_lats,50)} | {p(b_lats,95)} | 1.000 |
| FAISS | {p(f_lats,50)} | {p(f_lats,95)} | {round(float(np.mean(f_rec)),3) if f_rec else 0} |
| Graph | {p(g_lats,50)} | {p(g_lats,95)} | {round(float(np.mean(g_rec)),3) if g_rec else 0} |

**FAISS speedup vs Baseline: {round(p(b_lats,50)/p(f_lats,50),1) if p(f_lats,50)>0 else 'N/A'}x**
    """
    return result

# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

categories_list = ["All"] + sorted(hash_index.categories())
colors_list     = ["All"] + color_index.colors(None)
brands_list     = ["All"] + brand_index.top_brands(n=50)[0:50]
brands_list     = ["All"] + [b for b,_ in brand_index.top_brands(n=50)]
methods_list    = ["faiss","graph","baseline"]

def chat_with_assistant(message, history):
    """Gradio chatbot function."""
    if not message.strip():
        return history, [], "Please type a message."
    
    response, products = chatbot.chat(message)
    history = history or []
    history.append((message, response))
    
    # Format products for gallery
    gallery = []
    for p in products:
        img_path = f"{IMAGES_DIR}/{p['id']}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            caption = f"{p['name'][:30]}\n{p['type']} | {p['color']}\n🏷 {p['brand']}"
            gallery.append((img, caption))
    
    product_info = f"Found {len(products)} products" if products else ""
    return history, gallery, product_info

def reset_chat():
    chatbot.reset()
    return [], [], ""

CSS = """
.gradio-container { max-width: 1400px !important; font-family: 'Inter', sans-serif; }
.tab-nav button { font-size: 14px !important; font-weight: 600 !important; }
h1 { background: linear-gradient(90deg, #4FC3F7, #81C784);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2.5rem !important; font-weight: 900 !important; }
.result-info { background: #0d2137; border: 1px solid #1e4d3b;
               border-radius: 8px; padding: 10px 16px; color: #81C784; }
footer { display: none !important; }
"""

with gr.Blocks(
    css=CSS,
    title="FashionFinder v4.0",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
) as demo:

    gr.Markdown("""
# FashionFinder v4.0 Ultimate
### Visual Search & Image Retrieval System
**ResNet18 · k-NN Graph (444K edges) · FAISS ANN · MMR Diversity · Brand Compare · Style Transfer · Fashion Timeline**
*Built by Akila Lourdes Miriyala Francis & Akilan Manivannan · LIU Brooklyn · 2026*
---
    """)

    # ── Shared settings ────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            method_dd  = gr.Dropdown(methods_list, value="faiss",
                                      label="⚡ Retrieval Method",
                                      info="FAISS=fastest, Graph=scalable, Baseline=exact")
            k_sl       = gr.Slider(1, 20, value=10, step=1, label="Top-k Results")
            mmr_chk    = gr.Checkbox(value=False, label="🎨 MMR Diversity Reranking")
        with gr.Column(scale=1):
            cat_dd     = gr.Dropdown(categories_list, value="All", label="Category Filter")
            color_dd   = gr.Dropdown(colors_list,     value="All", label="Color Filter")
            brand_dd   = gr.Dropdown(brands_list,     value="All", label="Brand Filter")
        with gr.Column(scale=1):
            gr.Markdown(f"""
### 📊 System Info
- **Products:** {len(embeddings):,}
- **Brands:** {len(brand_index.all_brands())}
- **Graph edges:** 444,190
- **FAISS clusters:** 100
- **Embedding dims:** 512
            """)

    gr.Markdown("---")

    # ── TABS ───────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── TAB 1: IMAGE SEARCH ────────────────────────────────────────────────
        with gr.Tab("🔍 Image Search"):
            gr.Markdown("### Upload any fashion image to find visually similar products")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Fashion Image", type="pil", height=300)
                    search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                with gr.Column(scale=3):
                    img_info    = gr.Markdown("Results will appear here...")
                    img_gallery = gr.Gallery(label="Similar Products", columns=5,
                                             height=500, object_fit="contain")
            search_btn.click(
                fn=search_by_image,
                inputs=[img_input, method_dd, k_sl, mmr_chk, cat_dd, color_dd, brand_dd],
                outputs=[img_gallery, img_info]
            )

        # ── TAB 2: PRODUCT ID SEARCH ───────────────────────────────────────────
        with gr.Tab("🔢 Product ID"):
            gr.Markdown("### Search by product ID from the dataset")
            with gr.Row():
                with gr.Column(scale=1):
                    pid_input = gr.Number(value=1163, label="Product ID",
                                           info="Try: 1163 (cricket jersey), 2213 (dress), 1730 (shoes)")
                    pid_btn   = gr.Button("🔍 Search by ID", variant="primary", size="lg")
                with gr.Column(scale=3):
                    pid_info    = gr.Markdown("Results will appear here...")
                    pid_gallery = gr.Gallery(label="Similar Products", columns=5,
                                             height=500, object_fit="contain")
            pid_btn.click(
                fn=search_by_id,
                inputs=[pid_input, method_dd, k_sl, mmr_chk, cat_dd, color_dd, brand_dd],
                outputs=[pid_gallery, pid_info]
            )

        # ── TAB 3: URL SEARCH ──────────────────────────────────────────────────
        with gr.Tab("🌐 URL Search"):
            gr.Markdown("### Paste any image URL from the internet")
            with gr.Row():
                with gr.Column(scale=1):
                    url_input    = gr.Textbox(label="Image URL",
                                              placeholder="https://example.com/jacket.jpg")
                    url_btn      = gr.Button("🔍 Search URL", variant="primary", size="lg")
                    url_preview  = gr.Image(label="Preview", height=200)
                with gr.Column(scale=3):
                    url_info    = gr.Markdown("Results will appear here...")
                    url_gallery = gr.Gallery(label="Similar Products", columns=5,
                                             height=500, object_fit="contain")
            url_btn.click(
                fn=search_by_url,
                inputs=[url_input, method_dd, k_sl, mmr_chk],
                outputs=[url_preview, url_gallery, url_info]
            )

        # ── TAB 4: STYLE TRANSFER ──────────────────────────────────────────────
        with gr.Tab("🎨 Style Transfer"):
            gr.Markdown("### Find the same style in a completely different color. Unique to FashionFinder.")
            with gr.Row():
                with gr.Column(scale=1):
                    st_pid   = gr.Number(value=1163, label="Product ID")
                    st_color = gr.Dropdown(colors_list[1:], value="Red", label="Target Color")
                    st_k     = gr.Slider(1, 10, value=6, step=1, label="Results")
                    st_btn   = gr.Button("🎨 Transfer Style", variant="primary", size="lg")
                with gr.Column(scale=3):
                    st_info    = gr.Markdown("Results will appear here...")
                    st_gallery = gr.Gallery(label="Same Style, New Color", columns=5,
                                            height=500, object_fit="contain")
            st_btn.click(
                fn=search_style_transfer,
                inputs=[st_pid, st_color, st_k],
                outputs=[st_gallery, st_info]
            )

        # ── TAB 5: CLIP STYLE SEARCH ───────────────────────────────────────────
        with gr.Tab("🤖 Style Modifier"):
            gr.Markdown("### Upload image + pick a style modifier to transform the search")
            with gr.Row():
                with gr.Column(scale=1):
                    clip_img = gr.Image(label="Upload Image", type="pil", height=250)
                    clip_style = gr.Radio(
                        ["formal","casual","sporty","elegant","party","minimal","colorful","vintage","winter"],
                        value="elegant", label="Style Modifier"
                    )
                    clip_custom = gr.Textbox(label="Or type custom modifier",
                                             placeholder="e.g. 'office wear', 'beach outfit'")
                    clip_k   = gr.Slider(1, 20, value=10, step=1, label="Results")
                    clip_btn = gr.Button("🤖 Style Search", variant="primary", size="lg")
                with gr.Column(scale=3):
                    clip_info    = gr.Markdown("Results will appear here...")
                    clip_gallery = gr.Gallery(label="Style Results", columns=5,
                                              height=500, object_fit="contain")
            clip_btn.click(
                fn=search_clip_style,
                inputs=[clip_img, clip_style, clip_custom, clip_k],
                outputs=[clip_gallery, clip_info]
            )

        # ── TAB 6: BENCHMARK ───────────────────────────────────────────────────
        with gr.Tab("📊 Benchmark"):
            gr.Markdown("### Live benchmark — all 3 retrieval methods, real numbers")
            with gr.Row():
                bench_n   = gr.Slider(10, 200, value=50, step=10, label="Number of queries")
                bench_k   = gr.Slider(1, 20, value=10, step=1,  label="k (top results)")
                bench_btn = gr.Button("▶️ Run Benchmark", variant="primary")
            bench_out = gr.Markdown("Click Run Benchmark to see results...")
            bench_btn.click(fn=run_benchmark, inputs=[bench_n, bench_k], outputs=[bench_out])

        # ── TAB 7: AI CHATBOT ─────────────────────────────────────────────────────
        with gr.Tab("🤖 AI Fashion Assistant"):
            gr.Markdown("""
### Your Personal AI Fashion Shopping Assistant
Ask anything — find products, build outfits, get style advice, or ask about any occasion!

**Try:** *"What should I wear to a wedding?"* · *"Show me red Nike shoes"* · *"Build me a gym outfit"* · *"What goes with blue jeans?"*
            """)
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot_ui = gr.Chatbot(
                        label="FashionFinder AI Assistant",
                        height=450,
                        bubble_full_width=False,
                        avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"),
                        show_label=True,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask me anything... 'What should I wear to a party?', 'Show me blue formal shirts', 'Build me a winter outfit'",
                            label="Your message",
                            scale=4,
                            container=False,
                        )
                        chat_send = gr.Button("Send 💬", variant="primary", scale=1)
                    chat_reset = gr.Button("🔄 Reset Conversation", size="sm")
                    
                    # Quick question buttons
                    gr.Markdown("**Quick questions:**")
                    with gr.Row():
                        q1 = gr.Button("👰 Wedding outfit", size="sm")
                        q2 = gr.Button("💪 Gym outfit", size="sm")
                        q3 = gr.Button("💼 Office look", size="sm")
                        q4 = gr.Button("🎉 Party outfit", size="sm")
                    with gr.Row():
                        q5 = gr.Button("🏖️ Beach outfit", size="sm")
                        q6 = gr.Button("❄️ Winter outfit", size="sm")
                        q7 = gr.Button("🎓 College outfit", size="sm")
                        q8 = gr.Button("💕 Date night", size="sm")

                with gr.Column(scale=1):
                    chat_info    = gr.Markdown("Products mentioned in chat appear here...")
                    chat_gallery = gr.Gallery(
                        label="Products from Chat",
                        columns=2,
                        height=450,
                        object_fit="contain"
                    )

            # Wire up events
            chat_send.click(
                fn=chat_with_assistant,
                inputs=[chat_input, chatbot_ui],
                outputs=[chatbot_ui, chat_gallery, chat_info]
            ).then(lambda: "", outputs=[chat_input])

            chat_input.submit(
                fn=chat_with_assistant,
                inputs=[chat_input, chatbot_ui],
                outputs=[chatbot_ui, chat_gallery, chat_info]
            ).then(lambda: "", outputs=[chat_input])

            chat_reset.click(fn=reset_chat, outputs=[chatbot_ui, chat_gallery, chat_info])

            # Quick question buttons
            for btn, question in [
                (q1, "What should I wear to a wedding? I prefer traditional Indian outfits"),
                (q2, "Build me a complete gym workout outfit"),
                (q3, "What is a good professional office look for men?"),
                (q4, "Suggest a party outfit for a night out"),
                (q5, "What should I wear to the beach?"),
                (q6, "Build me a cozy winter outfit"),
                (q7, "What is a good college outfit for everyday wear?"),
                (q8, "Help me pick a date night outfit"),
            ]:
                btn.click(
                    fn=lambda q=question: chat_with_assistant(q, []),
                    outputs=[chatbot_ui, chat_gallery, chat_info]
                )

        # ── TAB 8: ABOUT ───────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## FashionFinder v4.0 — The Ultimate Edition

### What makes this unique vs Google Lens, Amazon, Pinterest?

| Feature | Google Lens | Amazon | Pinterest | **FashionFinder** |
|---------|-------------|--------|-----------|-------------------|
| Explainable similarity scores | ❌ | ❌ | ❌ | ✅ |
| Compare 3 retrieval algorithms live | ❌ | ❌ | ❌ | ✅ |
| Style Transfer (same style, new color) | ❌ | ❌ | ❌ | ✅ |
| Benchmark latency & recall yourself | ❌ | ❌ | ❌ | ✅ |
| 467 brand cross-comparison | ❌ | ❌ | ❌ | ✅ |
| k-NN graph visualization | ❌ | ❌ | ❌ | ✅ |
| Runs 100% transparently | ❌ | ❌ | ❌ | ✅ |

### Data Structures
- **Hash Table** `hash_index.py` — O(1) category lookup, ~70% space reduction
- **Nested Hash Table** `color_index.py` — O(1) category+color lookup
- **Brand Index** `brand_index.py` — 467 brands, 3-level nested hash
- **k-NN Graph** `knn_graph.py` — 444,190 edges, BFS traversal O(log n)
- **Min-Heap** `heap_ranker.py` — O(n log k) top-k ranking
- **FAISS IVF** `faiss_index.py` — 100 Voronoi clusters, sub-linear ANN

### Benchmark Results
| Method | Median Latency | Recall@10 |
|--------|---------------|-----------|
| Baseline (Brute-Force) | 8.63 ms | 1.000 |
| k-NN Graph | 9.37 ms | 0.894 |
| FAISS ANN | **1.79 ms** | **0.900** |

### Dataset
- **44,419** fashion products from Kaggle
- **7** master categories
- **47** colors
- **512-dim** ResNet18 embeddings (86 MB)

### Built by
**Akila Lourdes Miriyala Francis** & **Akilan Manivannan**
CS 631 — Algorithms and Data Structures · LIU Brooklyn · 2026

GitHub: [AKilalours/FashionFinder](https://github.com/AKilalours/FashionFinder)
            """)

demo.launch()
