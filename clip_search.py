"""
clip_search.py
--------------
CLIP Multimodal Search: Search with TEXT + IMAGE combined.

"Find a jacket like this but more formal"
"Show me something like this in a sporty style"
"Find this pattern but make it look elegant"

Uses OpenAI's CLIP (ViT-B/32) to encode both image and text,
then combines the vectors for multimodal retrieval.

This is state-of-the-art visual search — the same technique
used by Pinterest's visual search team.
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional, List, Tuple
import io


class CLIPSearch:
    """
    Multimodal search combining image embedding + text embedding.
    Uses CLIP ViT-B/32 for both image and text encoding.
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model     = None
        self.preprocess = None
        self._loaded   = False
        self._load_clip()

    def _load_clip(self):
        """Lazy-loads CLIP model."""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            self._loaded = True
            print(f"CLIP ViT-B/32 loaded on {self.device}")
        except ImportError:
            print("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
            self._loaded = False
        except Exception as e:
            print(f"CLIP load failed: {e}")
            self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded

    def encode_image(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """Encodes a PIL image to a 512-dim CLIP embedding."""
        if not self._loaded:
            return None
        import clip
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encodes a text prompt to a 512-dim CLIP embedding."""
        if not self._loaded:
            return None
        import clip
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze()

    def encode_combined(
        self,
        pil_image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        image_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> Optional[np.ndarray]:
        """
        Combines image and text embeddings into a single query vector.

        image_weight + text_weight should sum to 1.0
        Higher image_weight = results look more like the image
        Higher text_weight = results match the text description more
        """
        if not self._loaded:
            return None

        vecs = []
        weights = []

        if pil_image is not None:
            img_vec = self.encode_image(pil_image)
            if img_vec is not None:
                vecs.append(img_vec * image_weight)
                weights.append(image_weight)

        if text is not None and text.strip():
            txt_vec = self.encode_text(text)
            if txt_vec is not None:
                vecs.append(txt_vec * text_weight)
                weights.append(text_weight)

        if not vecs:
            return None

        combined = np.sum(vecs, axis=0)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def build_clip_embeddings(
        self,
        metadata,
        images_dir: str,
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> np.ndarray:
        """
        Builds CLIP embeddings for all products.
        These are separate from ResNet18 embeddings and enable
        text-based retrieval.

        Returns (N, 512) CLIP embedding matrix.
        """
        if not self._loaded:
            raise RuntimeError("CLIP not loaded.")

        import os
        from tqdm import tqdm

        rows = metadata.iterrows()
        if limit:
            import itertools
            rows = itertools.islice(rows, limit)

        all_vecs = []
        batch_imgs = []
        batch_idxs = []

        for idx, row in tqdm(rows, desc="Building CLIP embeddings"):
            pid = int(row.get("id", idx))
            img_path = os.path.join(images_dir, f"{pid}.jpg")
            if not os.path.exists(img_path):
                all_vecs.append(np.zeros(512, dtype=np.float32))
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                batch_imgs.append(self.preprocess(img))
                batch_idxs.append(len(all_vecs))
                all_vecs.append(None)  # placeholder
            except:
                all_vecs.append(np.zeros(512, dtype=np.float32))

            if len(batch_imgs) >= batch_size:
                self._process_batch(batch_imgs, batch_idxs, all_vecs)
                batch_imgs, batch_idxs = [], []

        if batch_imgs:
            self._process_batch(batch_imgs, batch_idxs, all_vecs)

        # Replace None placeholders with zeros
        result = []
        for v in all_vecs:
            if v is None:
                result.append(np.zeros(512, dtype=np.float32))
            else:
                result.append(v)

        return np.array(result, dtype=np.float32)

    def _process_batch(self, batch_imgs, batch_idxs, all_vecs):
        import clip
        stacked = torch.stack(batch_imgs).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(stacked)
            features = features / features.norm(dim=-1, keepdim=True)
        vecs = features.cpu().numpy()
        for i, idx in enumerate(batch_idxs):
            all_vecs[idx] = vecs[i]


# ── Predefined style modifier texts for the UI ──────────────────────────────

STYLE_MODIFIERS = {
    "More Formal":    "formal elegant professional office business attire",
    "More Casual":    "casual comfortable relaxed everyday streetwear",
    "More Sporty":    "athletic sports performance workout gym activewear",
    "More Elegant":   "luxury elegant sophisticated high fashion premium",
    "More Colorful":  "vibrant colorful bright bold striking vivid",
    "More Minimal":   "minimal simple clean monochrome understated",
    "Party Style":    "party evening glamorous festive celebration nightout",
    "Beach Style":    "beach summer tropical vacation resort swimwear",
    "Winter Style":   "winter warm cozy layered thermal cold weather",
    "Vintage Style":  "vintage retro classic old-school throwback heritage",
}
