"""
chatbot.py - FashionFinder AI Shopping Assistant
RAG (Retrieval Augmented Generation) + Text-to-Speech
"""

import os, re, json, random, requests
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

STYLE_MAP = {
    "formal":    ["Shirts","Formal Shoes","Trousers","Blazers","Ties","Belts"],
    "casual":    ["Tshirts","Jeans","Shorts","Casual Shoes","Tops","Flats"],
    "sporty":    ["Sports Shoes","Track Pants","Jackets","Sports Sandals","Socks"],
    "elegant":   ["Sarees","Kurtas","Gowns","Lehenga Cholis","Heels","Dupattas"],
    "party":     ["Dresses","Heels","Tops","Tunics","Skirts","Clutches","Earrings"],
    "wedding":   ["Sarees","Lehenga Cholis","Kurtas","Heels","Dupattas","Earrings"],
    "office":    ["Shirts","Formal Shoes","Trousers","Blazers","Belts","Ties"],
    "beach":     ["Shorts","Sandals","Tshirts","Sunglasses","Flats","Flip Flops"],
    "gym":       ["Sports Shoes","Track Pants","Tshirts","Sports Sandals","Socks"],
    "interview": ["Shirts","Formal Shoes","Trousers","Blazers","Ties","Belts"],
    "date":      ["Dresses","Tops","Heels","Jeans","Casual Shoes","Earrings"],
    "college":   ["Tshirts","Jeans","Casual Shoes","Backpacks","Shorts","Tops"],
    "festival":  ["Kurtas","Sarees","Lehenga Cholis","Heels","Dupattas","Earrings"],
    "dinner":    ["Dresses","Shirts","Heels","Trousers","Tops","Earrings"],
    "minimal":   ["Tshirts","Shirts","Trousers","Jeans","Flats","Sandals"],
    "colorful":  ["Tops","Shirts","Tshirts","Dresses","Kurtas","Skirts"],
    "vintage":   ["Jeans","Shirts","Jackets","Casual Shoes","Caps","Sunglasses"],
    "winter":    ["Jackets","Sweaters","Sweatshirts","Boots","Mufflers","Gloves"],
}

SYNONYMS = {
    "professional":"office","business":"office","work":"office","smart":"formal",
    "relaxed":"casual","comfortable":"casual","everyday":"casual",
    "athletic":"gym","workout":"gym","sport":"sporty","running":"sporty",
    "luxury":"elegant","sophisticated":"elegant","traditional":"wedding",
    "ethnic":"festival","saree":"wedding","kurta":"festival",
    "night out":"party","clubbing":"party","celebration":"party",
    "simple":"minimal","plain":"minimal","basic":"minimal",
    "bright":"colorful","vibrant":"colorful","bold":"colorful",
    "retro":"vintage","classic":"vintage","throwback":"vintage",
    "cold":"winter","cozy":"winter","warm":"winter",
    "summer":"beach","tropical":"beach","holiday":"beach",
}

COLOR_WORDS = [
    "red","blue","green","black","white","yellow","pink","purple",
    "orange","brown","grey","gray","navy","beige","cream","maroon",
    "gold","silver","copper","teal","olive","coral","turquoise","lavender"
]

BRAND_KEYWORDS = [
    "nike","adidas","puma","reebok","levis","wrangler","allen solly",
    "van heusen","fabindia","biba","only","vero moda","jack jones",
    "peter england","arrow","raymond","woodland","bata","liberty",
    "fastrack","tantra","gini","jony","scullers","locomotive","spykar",
    "flying machine","pepe jeans","united colors","benetton"
]

ARTICLE_HINTS = {
    "shoes":["Sports Shoes","Casual Shoes","Formal Shoes","Heels","Sandals","Flats","Boots"],
    "shirt":["Shirts","Tshirts","Polo Tshirts","Formal Shirts"],
    "tshirt":["Tshirts"],"t-shirt":["Tshirts"],
    "jeans":["Jeans"],"trouser":["Trousers"],
    "dress":["Dresses","Ethnic Dress"],"saree":["Sarees"],"kurta":["Kurtas"],
    "jacket":["Jackets"],"bag":["Handbags","Backpacks","Clutches"],
    "watch":["Watches"],"sunglass":["Sunglasses"],
    "sandal":["Sandals","Sports Sandals","Flip Flops"],
    "heels":["Heels"],"sneaker":["Sports Shoes","Casual Shoes"],
    "shorts":["Shorts"],"skirt":["Skirts"],"top":["Tops","Tunics"],
    "earring":["Earrings"],"belt":["Belts"],"tie":["Ties"],"blazer":["Blazers"],
    "sweater":["Sweaters","Sweatshirts"],"hoodie":["Sweatshirts"],
    "track":["Track Pants"],"legging":["Leggings"],
}


class RAGFashionChatbot:
    def __init__(self, metadata, embeddings, hash_index, color_index,
                 brand_index, heap_ranker_fn, embedder, faiss_index=None, api_key=None):
        self.metadata     = metadata
        self.embeddings   = embeddings
        self.hash_index   = hash_index
        self.color_index  = color_index
        self.brand_index  = brand_index
        self.top_k_cosine = heap_ranker_fn
        self.embedder     = embedder
        self.faiss_index  = faiss_index
        self.api_key      = api_key or os.environ.get("ANTHROPIC_API_KEY","")
        self.messages     = []
        self.last_products= []
        self._all_colors  = color_index.colors(None)
        self._color_lower = {c.lower(): c for c in self._all_colors}
        self._all_brands  = brand_index.all_brands()

    # ── Intent parsing ─────────────────────────────────────────────────────────
    def _parse_intent(self, msg: str) -> Dict:
        ml = msg.lower()
        style = next((k for k in STYLE_MAP if k in ml), None)
        if not style:
            style = next((SYNONYMS[s] for s in SYNONYMS if s in ml), None)
        colors  = [c for c in COLOR_WORDS if c in ml]
        brands  = [b for bk in BRAND_KEYWORDS if bk in ml
                   for b in [next((x for x in self._all_brands if bk.lower() in x.lower()), None)]
                   if b]
        pid_m   = re.search(r'\b(\d{4,6})\b', msg)
        articles= list({t for k,types in ARTICLE_HINTS.items() if k in ml for t in types})
        kind    = ("greeting"    if any(w in ml for w in ["hello","hi","hey","help","who are you"]) else
                   "outfit"      if any(w in ml for w in ["outfit","what to wear","what should i wear","what goes with","pair","match"]) else
                   "search"      if any(w in ml for w in ["show","find","search","give me","i want","recommend","suggest"]) else
                   "product_info"if pid_m else
                   "advice"      if any(w in ml for w in ["advice","tip","how to","style"]) else
                   "general")
        return {"type":kind,"style":style,"colors":colors,"brands":brands,
                "pid":int(pid_m.group(1)) if pid_m else None,"articles":articles,"raw":ml}

    def _resolve_color(self, c: str) -> Optional[str]:
        if not c: return None
        cl = c.lower()
        if cl in self._color_lower: return self._color_lower[cl]
        return next((v for k,v in self._color_lower.items() if cl in k or k in cl), None)

    # ── RAG Step 1: Retrieve ───────────────────────────────────────────────────
    def _retrieve(self, intent: Dict, k: int = 6) -> List[Dict]:
        if intent["pid"]:
            m = self.metadata[self.metadata["id"] == intent["pid"]]
            if not m.empty:
                return [self._to_dict(m.index[0], m.iloc[0], 1.0)]

        cands = list(range(len(self.embeddings)))

        # Brand filter
        if intent["brands"]:
            bc = []
            for b in intent["brands"]:
                bc.extend(self.brand_index.get_indices(brand=b))
            if bc: cands = list(set(bc))

        # Color filter
        if intent["colors"]:
            rc = self._resolve_color(intent["colors"][0])
            if rc:
                cc = self.color_index.get_indices(color=rc)
                if cc: cands = [i for i in cands if i in set(cc)]

        # Article type filter
        target_types = intent["articles"] or (STYLE_MAP.get(intent["style"], []) if intent["style"] else [])
        if target_types:
            tl = set(t.lower() for t in target_types)
            fc = [i for i in cands if str(self.metadata.loc[i].get("articleType","")).lower() in tl]
            if len(fc) >= k: cands = fc

        random.shuffle(cands)
        seen = set(); diverse = []
        for i in cands:
            art = str(self.metadata.loc[i].get("articleType",""))
            if art not in seen or len(diverse) < k:
                diverse.append(i); seen.add(art)
            if len(diverse) >= k: break

        return [self._to_dict(i, self.metadata.loc[i], 0.85) for i in diverse[:k]]

    def _retrieve_outfit(self, style: str, color: str = None) -> List[Dict]:
        types = STYLE_MAP.get(style, STYLE_MAP["casual"])
        outfit = []
        for art in types[:6]:
            cands = list(range(len(self.embeddings)))
            if color:
                rc = self._resolve_color(color)
                if rc:
                    cc = self.color_index.get_indices(color=rc)
                    if cc: cands = cc
            tc = [i for i in cands if str(self.metadata.loc[i].get("articleType","")).lower() == art.lower()]
            if tc:
                idx = random.choice(tc[:50])
                item = self._to_dict(idx, self.metadata.loc[idx], 0.9)
                item["outfit_role"] = art
                outfit.append(item)
        return outfit

    def _to_dict(self, idx, row, score) -> Dict:
        pid = int(row.get("id", idx))
        return {"id":pid,"idx":int(idx),
                "name":  str(row.get("productDisplayName","Unknown")),
                "type":  str(row.get("articleType","Unknown")),
                "color": str(row.get("baseColour","Unknown")),
                "category":str(row.get("masterCategory","Unknown")),
                "season":str(row.get("season","Unknown")),
                "brand": self.brand_index.get_brand(idx),
                "score": round(score,3)}

    # ── RAG Step 2: Augment context ────────────────────────────────────────────
    def _build_context(self, products: List[Dict]) -> str:
        if not products: return ""
        lines = [f"RETRIEVED PRODUCTS ({len(products)} items from 44,419 catalog):"]
        for i,p in enumerate(products,1):
            role = f"[{p.get('outfit_role','')}] " if "outfit_role" in p else ""
            lines.append(f"{i}. {role}ID:{p['id']} | {p['name'][:50]} | "
                         f"Type:{p['type']} | Color:{p['color']} | Brand:{p['brand']}")
        return "\n".join(lines)

    # ── RAG Step 3: Generate via Claude ───────────────────────────────────────
    def _call_claude(self, user_msg: str, context: str) -> Optional[str]:
        if not self.api_key: return None
        aug = user_msg + (f"\n\n[RAG CONTEXT]\n{context}" if context else "")
        msgs = self.messages[-8:] + [{"role":"user","content":aug}]
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key":self.api_key,
                         "anthropic-version":"2023-06-01",
                         "content-type":"application/json"},
                json={"model":"claude-sonnet-4-20250514","max_tokens":512,
                      "system":self._system(),"messages":msgs},
                timeout=30
            )
            if r.status_code == 200:
                return r.json()["content"][0]["text"]
        except Exception as e:
            print(f"Claude error: {e}")
        return None

    def _system(self) -> str:
        return """You are FashionFinder's AI Shopping Assistant — friendly, expert, concise.
You use RAG: real products from a 44,419-item catalog are retrieved and given to you.
RULES:
- Only mention products from the RAG CONTEXT. Never invent products.
- Always include product IDs so users can find them.
- Be warm, specific, helpful. Use emojis naturally.
- Keep responses under 5 sentences unless building a full outfit.
- For outfits: explain why each piece works together.
- For searches: highlight 2-3 best options with names and IDs.
- For advice: give a fashion tip then connect to retrieved products."""

    def _fallback(self, intent: Dict, products: List[Dict]) -> str:
        t = intent["type"]
        if t == "greeting":
            return ("Hi! I'm FashionFinder's AI assistant! 👋\n"
                    "Ask me to find products, build outfits, or give style advice.\n"
                    "Try: 'What to wear to a wedding?' or 'Show me red Nike shoes'!")
        if products:
            if t == "outfit":
                items = [f"{p['type']} from {p['brand']}" for p in products[:4]]
                return f"Here's your {intent['style'] or 'complete'} outfit! 👗 {', '.join(items)}. Check the cards below!"
            if t in ("search","product_info"):
                p = products[0]
                return f"Found {len(products)} items! Top pick: {p['name'][:40]} (ID:{p['id']}, Brand:{p['brand']}). Check below!"
        return ("Try asking:\n• 'Outfit for a party'\n• 'Red Nike shoes'\n"
                "• 'Build a gym outfit'\n• 'What to wear to a wedding?'")

    # ── RAG Step 4: TTS JavaScript ────────────────────────────────────────────
    def get_tts_js(self, text: str) -> str:
        """Browser Web Speech API — no API key needed."""
        clean = re.sub(r'\*+|#+\s*', '', text)
        clean = re.sub(r'[^\w\s.,!?\'()-]', '', clean)
        clean = re.sub(r'\n+', '. ', clean).strip()[:400]
        clean = clean.replace('"', "'").replace('\\','')
        return f"""
<script>
(function(){{
  if(!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  var u = new SpeechSynthesisUtterance("{clean}");
  u.rate=0.95; u.pitch=1.05; u.volume=1.0; u.lang='en-US';
  function speak(){{
    var voices = window.speechSynthesis.getVoices();
    var v = voices.find(v=>v.name.includes('Samantha')||
                            v.name.includes('Google UK English Female')||
                            v.name.includes('Female')||
                            v.lang==='en-US');
    if(v) u.voice=v;
    window.speechSynthesis.speak(u);
  }}
  if(window.speechSynthesis.getVoices().length>0) speak();
  else window.speechSynthesis.addEventListener('voiceschanged', speak, {{once:true}});
}})();
</script>"""

    # ── Main chat function ─────────────────────────────────────────────────────
    def chat(self, user_message: str) -> Tuple[str, List[Dict], str]:
        """
        Returns (response_text, products_list, tts_javascript)
        """
        intent = self._parse_intent(user_message)

        # Retrieve
        if intent["type"] == "outfit":
            products = self._retrieve_outfit(intent["style"] or "casual",
                                             intent["colors"][0] if intent["colors"] else None)
        elif intent["type"] in ("search","product_info","advice"):
            products = self._retrieve(intent, k=6)
        else:
            products = []

        # Augment
        context = self._build_context(products)

        # Generate
        response = self._call_claude(user_message, context) or self._fallback(intent, products)

        # Update history
        self.messages.append({"role":"user","content":user_message})
        self.messages.append({"role":"assistant","content":response})
        self.last_products = products

        return response, products, self.get_tts_js(response)

    def reset(self):
        self.messages = []
        self.last_products = []
