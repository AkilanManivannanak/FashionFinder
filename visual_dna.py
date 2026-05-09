"""
visual_dna.py
-------------
Visual DNA: Shows what ResNet18 actually saw in an image.

Uses GradCAM (Gradient-weighted Class Activation Mapping) to highlight
which parts of the image contributed most to the embedding.
Also generates a 512-dim embedding visualization as a heatmap.

No extra training needed — uses the existing ResNet18 backbone.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class VisualDNA:
    """
    Generates visual explanations of what ResNet18 sees in an image.
    Uses GradCAM on the last convolutional layer.
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Full ResNet18 with hooks for GradCAM
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Hooks for GradCAM
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hook on last conv layer of ResNet18 (layer4)
        target_layer = self.model.layer4[-1].conv2
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, pil_image: Image.Image) -> np.ndarray:
        """
        Generates GradCAM heatmap for the input image.
        Returns a (224, 224) numpy array with attention weights.
        """
        img = pil_image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        # Forward pass
        output = self.model(tensor)

        # Use the max activation as the "class" to explain
        score = output[0].max()
        self.model.zero_grad()
        score.backward()

        # GradCAM computation
        gradients  = self.gradients[0]   # (512, 7, 7)
        activations = self.activations[0] # (512, 7, 7)

        weights = gradients.mean(dim=(1, 2))  # (512,)
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize and resize to 224x224
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to image size
        from PIL import Image as PILImage
        cam_img = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((224, 224), PILImage.BILINEAR)
        cam_np  = np.array(cam_img) / 255.0

        return cam_np

    def generate_embedding_heatmap(self, embedding: np.ndarray) -> bytes:
        """
        Visualizes the 512-dim embedding vector as a 32x16 heatmap.
        Shows which embedding dimensions are most activated.
        Returns PNG bytes.
        """
        emb_2d = embedding.reshape(32, 16)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        im = ax.imshow(emb_2d, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Activation strength")
        ax.set_title("Visual DNA — 512-dim ResNet18 Embedding", color="white", fontsize=12)
        ax.set_xlabel("Embedding dimensions (0-15)", color="white")
        ax.set_ylabel("Embedding dimensions (16-31)", color="white")
        ax.tick_params(colors="white")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#0e1117", dpi=100)
        plt.close()
        buf.seek(0)
        return buf.read()

    def generate_attention_overlay(self, pil_image: Image.Image) -> bytes:
        """
        Overlays the GradCAM attention map on the original image.
        Red areas = ResNet18 paid most attention here.
        Returns PNG bytes.
        """
        img = pil_image.convert("RGB").resize((224, 224))
        cam = self.generate_gradcam(pil_image)

        # Create colored heatmap
        heatmap = cm.jet(cam)[:, :, :3]  # (224, 224, 3)
        img_np  = np.array(img) / 255.0  # (224, 224, 3)

        # Blend
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0e1117")
        titles = ["Original Image", "Attention Map\n(What ResNet18 saw)", "Overlay\n(Combined)"]
        images = [img_np, cm.jet(cam)[:, :, :3], overlay]

        for ax, title, image in zip(axes, titles, images):
            ax.imshow(image)
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")
            ax.set_facecolor("#0e1117")

        plt.suptitle("Visual DNA — Attention Analysis", color="#4FC3F7",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#0e1117", dpi=120)
        plt.close()
        buf.seek(0)
        return buf.read()

    def top_activated_regions(self, cam: np.ndarray, n: int = 5) -> list:
        """Returns the top-n most attended regions as (y, x, strength) tuples."""
        flat = cam.flatten()
        top_idx = np.argsort(flat)[-n:][::-1]
        results = []
        for idx in top_idx:
            y, x = divmod(idx, cam.shape[1])
            results.append({
                "y": int(y), "x": int(x),
                "strength": round(float(flat[idx]), 3),
                "region": _region_name(y, x, cam.shape)
            })
        return results


def _region_name(y: int, x: int, shape: tuple) -> str:
    """Maps pixel coordinates to human-readable region name."""
    h, w = shape
    vert = "top" if y < h//3 else ("middle" if y < 2*h//3 else "bottom")
    horiz = "left" if x < w//3 else ("center" if x < 2*w//3 else "right")
    return f"{vert}-{horiz}"
