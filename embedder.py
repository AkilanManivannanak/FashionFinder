"""
engine/embedder.py
------------------
Extracts 512-dim visual embeddings from fashion product images
using a pretrained ResNet18 (ImageNet weights, final FC layer removed).

Usage:
    from engine.embedder import Embedder
    embedder = Embedder()
    vector = embedder.embed_image("path/to/image.jpg")   # shape: (512,)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os


class Embedder:
    """
    Wraps a pretrained ResNet18 backbone.
    Removes the final classification layer so the output
    is a 512-dim feature vector per image.
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load pretrained ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final FC classification layer
        # Output is now the 512-dim avg pool feature vector
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Embedder ready on {self.device}")

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Takes a file path, returns a normalized 512-dim numpy vector.
        Returns None if the image cannot be loaded.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            return None

        tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            feat = self.model(tensor)              # (1, 512, 1, 1)
            feat = feat.squeeze().cpu().numpy()    # (512,)

        # L2 normalize so cosine similarity = dot product
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat  # shape: (512,)

    def embed_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Takes a PIL Image directly (used by FastAPI upload endpoint).
        Returns a normalized 512-dim numpy vector.
        """
        img = pil_image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat.squeeze().cpu().numpy()

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat
