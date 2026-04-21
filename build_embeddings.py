"""
build_embeddings.py
-------------------
Run this ONCE after downloading the Kaggle dataset.
Reads every image, extracts a 512-dim ResNet embedding,
and saves two files:

    embeddings/embeddings.npy      shape: (N, 512)
    embeddings/metadata.csv        columns: id, filename, productDisplayName,
                                            masterCategory, subCategory,
                                            articleType, baseColour, season, year

Usage:
    python build_embeddings.py --images_dir data/fashion-dataset/images
                               --styles_csv  data/fashion-dataset/styles.csv
                               --out_dir     embeddings/
                               --limit       5000   (optional, for quick testing)
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from embedder import Embedder


def build(images_dir: str, styles_csv: str, out_dir: str, limit: int = None):
    os.makedirs(out_dir, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────────
    print("Loading metadata...")
    df = pd.read_csv(styles_csv, on_bad_lines="skip")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    if limit:
        df = df.head(limit)
        print(f"Limiting to {limit} products for quick test")

    print(f"Total products in metadata: {len(df)}")

    # ── Match images that actually exist on disk ──────────────────────────────
    valid_rows = []
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, f"{int(row['id'])}.jpg")
        if os.path.exists(img_path):
            valid_rows.append(row)

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"Images found on disk: {len(df_valid)}")

    if len(df_valid) == 0:
        print("ERROR: No images found. Check your --images_dir path.")
        return

    # ── Extract embeddings ───────────────────────────────────────────────────
    embedder = Embedder()
    embeddings = []
    good_rows = []

    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Embedding"):
        img_path = os.path.join(images_dir, f"{int(row['id'])}.jpg")
        vec = embedder.embed_image(img_path)
        if vec is not None:
            embeddings.append(vec)
            good_rows.append(row)

    embeddings = np.array(embeddings, dtype=np.float32)  # (N, 512)
    df_out = pd.DataFrame(good_rows).reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    emb_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "metadata.csv")

    np.save(emb_path, embeddings)
    df_out.to_csv(meta_path, index=False)

    print(f"\nDone.")
    print(f"  Embeddings : {emb_path}  shape={embeddings.shape}")
    print(f"  Metadata   : {meta_path}  rows={len(df_out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="data/fashion-dataset/images")
    parser.add_argument("--styles_csv",  default="data/fashion-dataset/styles.csv")
    parser.add_argument("--out_dir",     default="embeddings")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Cap number of products (useful for quick testing)")
    args = parser.parse_args()

    build(args.images_dir, args.styles_csv, args.out_dir, args.limit)
