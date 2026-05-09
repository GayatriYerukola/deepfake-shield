"""
Grad-CAM Explainability for Vision Transformer (ViT)
=====================================================
Produces a heatmap showing which image regions the model considers
most suspicious — making the "black box" decision interpretable.

How Grad-CAM works for ViT
---------------------------
1. Register forward + backward hooks on the last encoder layer.
2. Run a forward pass → save patch-token activations.
3. Run a backward pass for the predicted class → save gradients.
4. Weight activations by their gradients (importance = gradient magnitude).
5. Apply ReLU (keep only positive contributions).
6. Reshape from (num_patches,) → (14×14) grid → upsample to image size.

Output: a heatmap where RED = most suspicious, BLUE = least suspicious.

Reference
---------
Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" ICCV 2017.
Dosovitskiy et al. "An Image is Worth 16×16 Words: ViT" ICLR 2021.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional


# ── Core Grad-CAM engine ──────────────────────────────────────────────────────

class GradCAMViT:
    """
    Gradient-weighted Class Activation Mapping for HuggingFace ViT models.

    Usage
    -----
    cam_engine = GradCAMViT(vit_model)
    heatmap    = cam_engine.run(image_tensor)   # numpy array 0-1
    cam_engine.remove_hooks()
    """

    def __init__(self, model):
        self._model      = model
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        # Hook onto the last encoder layer's first LayerNorm.
        # This sits just before the attention block output — rich spatial info.
        target = model.vit.encoder.layer[-1].layernorm_before
        self._fwd = target.register_forward_hook(self._save_acts)
        self._bwd = target.register_full_backward_hook(self._save_grads)

    # ── Hooks ────────────────────────────────────────────────────────────────

    def _save_acts(self, _mod, _inp, output):
        # output: (batch, num_patches + 1, hidden_dim)  +1 for [CLS]
        self._acts = output.detach()

    def _save_grads(self, _mod, _inp, grad_output):
        self._grads = grad_output[0].detach()

    # ── Main method ───────────────────────────────────────────────────────────

    def run(
        self,
        pixel_values: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        pixel_values : torch.Tensor  shape (1, 3, H, W), preprocessed by ViTImageProcessor
        target_class : int | None    class to explain; None → predicted class

        Returns
        -------
        np.ndarray  float32, shape (H, W), values in [0, 1]
        """
        self._model.eval()

        # ── Forward pass ──────────────────────────────────────────────────
        pixel_values = pixel_values.clone().requires_grad_(True)
        output  = self._model(pixel_values=pixel_values)
        logits  = output.logits                         # (1, num_classes)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # ── Backward pass for target class ────────────────────────────────
        self._model.zero_grad()
        logits[0, target_class].backward()

        # ── Build CAM from hooks ──────────────────────────────────────────
        acts  = self._acts   # (1, N+1, D)
        grads = self._grads  # (1, N+1, D)

        if acts is None or grads is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # Drop [CLS] token (index 0)
        acts  = acts[:, 1:, :]   # (1, N, D)
        grads = grads[:, 1:, :]  # (1, N, D)

        # Global-average-pool gradients over the hidden dimension → weights per patch
        weights = grads.mean(dim=2)              # (1, N)

        # Weighted combination of activations
        cam = (weights.unsqueeze(-1) * acts).sum(dim=2)  # (1, N)
        cam = torch.relu(cam.squeeze(0))                  # (N,)  ReLU: keep positive

        # Reshape to square grid (ViT-base: 14×14 for 224px / 16px patches)
        n_patches = int(cam.shape[0] ** 0.5)
        cam_grid  = cam[:n_patches * n_patches].reshape(n_patches, n_patches)
        cam_np    = cam_grid.detach().numpy().astype(np.float32)

        # Normalize to [0, 1]
        lo, hi = cam_np.min(), cam_np.max()
        if hi - lo > 1e-8:
            cam_np = (cam_np - lo) / (hi - lo)

        # Upsample to input image size
        img_h = pixel_values.shape[2]
        img_w = pixel_values.shape[3]
        cam_full = cv2.resize(cam_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        return cam_full

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


# ── Attention Rollout (fallback — no gradients needed) ────────────────────────

def attention_rollout(model, pixel_values: torch.Tensor) -> np.ndarray:
    """
    Attention Rollout: simpler alternative to Grad-CAM that requires no gradients.
    Works by multiplying attention matrices across all layers.

    Returns np.ndarray float32, same spatial shape as pixel_values.

    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" ACL 2020.
    """
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=pixel_values, output_attentions=True)

    attn_maps = out.attentions   # tuple of (1, heads, N+1, N+1) per layer

    # Average over heads, then apply rollout
    rollout = torch.eye(attn_maps[0].shape[-1])   # identity start

    for attn in attn_maps:
        avg   = attn.mean(dim=1).squeeze(0)       # (N+1, N+1)
        # Add identity residual (accounts for skip connections)
        avg   = avg + torch.eye(avg.shape[0])
        avg   = avg / avg.sum(dim=-1, keepdim=True)
        rollout = avg @ rollout

    # [CLS] token attends to all patches → row 0, columns 1:
    cls_attn  = rollout[0, 1:].numpy()             # (N,)
    n         = int(cls_attn.shape[0] ** 0.5)
    cam_grid  = cls_attn.reshape(n, n).astype(np.float32)

    lo, hi = cam_grid.min(), cam_grid.max()
    if hi - lo > 1e-8:
        cam_grid = (cam_grid - lo) / (hi - lo)

    img_h = pixel_values.shape[2]
    img_w = pixel_values.shape[3]
    return cv2.resize(cam_grid, (img_w, img_h), interpolation=cv2.INTER_LINEAR)


# ── Overlay helper ────────────────────────────────────────────────────────────

def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over the original image using JET colormap.

    Parameters
    ----------
    original_rgb : np.ndarray  uint8 RGB image
    heatmap      : np.ndarray  float32 [0-1], must match image spatial size
    alpha        : float       heatmap opacity (0 = invisible, 1 = opaque)

    Returns
    -------
    np.ndarray  uint8 RGB blended image
    """
    h, w = original_rgb.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = (
        (1 - alpha) * original_rgb.astype(np.float32)
        + alpha      * colored.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return blended


def highlight_top_regions(heatmap: np.ndarray, top_pct: float = 0.15) -> np.ndarray:
    """
    Create a binary mask of the top N% most suspicious pixels.
    Useful for drawing bounding boxes around flagged regions.
    """
    threshold = np.percentile(heatmap, (1 - top_pct) * 100)
    return (heatmap >= threshold).astype(np.uint8)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def generate_explanation(image_path: str) -> dict:
    """
    End-to-end explainability pipeline.

    Tries Grad-CAM first; falls back to Attention Rollout if gradients fail.

    Returns
    -------
    dict
        heatmap      : np.ndarray float32 (H, W)  — raw heatmap 0-1
        overlay      : np.ndarray uint8   (H, W, 3) — blended image
        original     : np.ndarray uint8   (H, W, 3) — resized original
        mask         : np.ndarray uint8   (H, W)   — top-15% binary mask
        method       : str  "gradcam" | "attention_rollout" | "unavailable"
        fake_score   : float | None
        top_region   : str  human-readable location of most suspicious region
        error        : str | None
    """
    try:
        from detector.model_manager import model_manager

        if not model_manager.is_loaded:
            return _err("Model not loaded. Load it from the sidebar first.")

        # Access the underlying ViT model and processor via public properties
        vit   = model_manager.vit_model
        proc  = model_manager.image_processor

        img_pil = Image.open(image_path).convert("RGB")
        inputs  = proc(images=img_pil, return_tensors="pt")
        tensor  = inputs["pixel_values"]               # (1, 3, H, W)

        img_size = tensor.shape[2]                      # usually 224

        # Resize original for overlay (matches model input)
        original_np = np.array(img_pil.resize((img_size, img_size)))

        # ── Try Grad-CAM ──────────────────────────────────────────────────
        method = "gradcam"
        try:
            engine  = GradCAMViT(vit)
            heatmap = engine.run(tensor)
            engine.remove_hooks()
        except Exception:
            # Fallback: attention rollout (no gradients needed)
            method  = "attention_rollout"
            heatmap = attention_rollout(vit, tensor)

        # ── Build overlay ─────────────────────────────────────────────────
        overlay = overlay_heatmap(original_np, heatmap)
        mask    = highlight_top_regions(heatmap)

        # ── Get prediction score ──────────────────────────────────────────
        pred    = model_manager.predict(image_path)
        fscore  = pred["fake_score"]

        # ── Describe most suspicious region ──────────────────────────────
        top_region = _describe_region(heatmap)

        return {
            "heatmap":    heatmap,
            "overlay":    overlay,
            "original":   original_np,
            "mask":       mask,
            "method":     method,
            "fake_score": fscore,
            "top_region": top_region,
            "error":      None,
        }

    except Exception as exc:
        return _err(str(exc))


def _describe_region(heatmap: np.ndarray) -> str:
    """Return a human-readable description of where the hotspot is."""
    h, w  = heatmap.shape
    cy    = np.average(np.arange(h), weights=heatmap.mean(axis=1))
    cx    = np.average(np.arange(w), weights=heatmap.mean(axis=0))

    vert  = "upper" if cy < h * 0.4 else ("lower" if cy > h * 0.6 else "middle")
    horiz = "left"  if cx < w * 0.4 else ("right"  if cx > w * 0.6 else "center")

    return f"{vert}-{horiz} region"


def _err(msg: str) -> dict:
    return {
        "heatmap": None, "overlay": None, "original": None,
        "mask": None, "method": "unavailable",
        "fake_score": None, "top_region": "unknown", "error": msg,
    }
