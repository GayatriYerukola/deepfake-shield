"""
FFT-CNN Hybrid Model — Custom Architecture
===========================================
This is YOUR model — designed and trained specifically for this project.

Architecture:
    CNN Branch  : ResNet18 pretrained on ImageNet → 512-dim spatial features
    FFT Branch  : Custom conv net on frequency domain → 128-dim frequency features
    Fusion      : Concatenate [512+128] → Linear(256) → Dropout(0.3) → 2 classes

Why two branches?
    - CNN branch detects visual artifacts (blending seams, texture inconsistencies)
    - FFT branch detects frequency-domain artifacts (GAN upsampling grid patterns)
    - Together they catch things neither branch can alone

Training results:
    Epoch 3 → val_acc = 99.29%  (best)
    Epoch 4 → val_acc = 99.09%
    Final   → val_acc ≈ 99.29%
"""

import torch
import torch.nn as nn
import torch.fft
import torchvision.models as models
from PIL import Image
import numpy as np


# ── Model definition (must match exactly what was used during training) ────────

def _fft_transform(x: torch.Tensor) -> torch.Tensor:
    """Convert image to log-magnitude FFT spectrum."""
    fft   = torch.fft.fft2(x, dim=(-2, -1))
    fft   = torch.fft.fftshift(fft)
    return torch.log1p(torch.abs(fft))


class FFTBranch(nn.Module):
    """Small CNN that processes the frequency-domain representation of an image."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), -1)


class FFT_CNN_Model(nn.Module):
    """
    Dual-branch deepfake detector.

    Input:  (B, 3, 224, 224) tensor, normalised with mean=0.5 std=0.5
    Output: (B, 2) logits  — index 0 = Real, index 1 = Fake
    """

    def __init__(self):
        super().__init__()

        # CNN branch — ResNet18, replace final FC with identity
        backbone    = models.resnet18(weights=None)   # weights loaded from .pth
        backbone.fc = nn.Identity()
        self.cnn    = backbone

        self.fft_branch = FFTBranch()

        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)
        fft_feat = self.fft_branch(_fft_transform(x))
        return self.classifier(torch.cat([cnn_feat, fft_feat], dim=1))


# Alias — keeps compatibility with any code using the camelCase name
FFTCNNModel = FFT_CNN_Model


# ── Image preprocessing (matches training transforms) ─────────────────────────

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image exactly as done during training.
    Returns tensor shape (1, 3, 224, 224).
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Same normalisation as albumentations Normalize(mean=0.5, std=0.5)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # add batch dim


# ── Loader ────────────────────────────────────────────────────────────────────

def load_fft_cnn_model(model_path: str, device: str = "cpu") -> FFT_CNN_Model:
    """
    Load the trained FFT-CNN model from disk.

    Supports:
        deepfake_model.pkl      — full model saved with torch.save(model, path)
        deepfake_model.pth.zip  — PyTorch zip format
        deepfake_model.pth      — state dict or full model
    """
    import sys, __main__

    # Inject class names so pickle can find them
    # (model was saved from a Jupyter notebook where classes lived in __main__)
    __main__.FFT_CNN_Model = FFT_CNN_Model
    __main__.FFTBranch     = FFTBranch
    sys.modules['__main__'].FFT_CNN_Model = FFT_CNN_Model
    sys.modules['__main__'].FFTBranch     = FFTBranch

    # Custom unpickler that remaps CUDA tensors → CPU
    # Needed because the model was pickled on a GPU machine
    import io, pickle

    class _CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location="cpu", weights_only=False
                )
            return super().find_class(module, name)

    with open(model_path, "rb") as f:
        loaded = _CPUUnpickler(f).load()

    if isinstance(loaded, FFT_CNN_Model):
        model = loaded
    elif isinstance(loaded, dict):
        model = FFT_CNN_Model()
        state = {k.replace("module.", ""): v for k, v in loaded.items()}
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unrecognised format: {type(loaded)}")

    model.to(device)
    model.eval()
    return model
