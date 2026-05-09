"""
Model Manager
=============
Handles lazy loading, caching, and inference for the HuggingFace
deepfake detection model.

To swap models change DEFAULT_MODEL_ID — nothing else needs to change.

Recommended models (all free, HuggingFace Hub):
  "dima806/deepfake_vs_real_image_detection"   ← default, ViT fine-tuned on faces
  "Wvolf/ViT-Deepfake-Detection"               ← alternate ViT
  "umm-maybe/AI-image-detector"                ← broader AI-image detection
  "Organika/sdxl-detector"                     ← best for Stable Diffusion images

First call downloads the model weights once (~85–300 MB depending on model).
All subsequent calls use the local HuggingFace cache (~/.cache/huggingface/).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Change this to switch models ──────────────────────────────────────────────
DEFAULT_MODEL_ID = "dima806/deepfake_vs_real_image_detection"


class ModelManager:
    """
    Singleton wrapper around a HuggingFace image-classification pipeline.

    Usage
    -----
    from detector.model_manager import model_manager

    success = model_manager.load()          # downloads once, then cached
    if model_manager.is_loaded:
        result = model_manager.predict("path/to/image.jpg")
        print(result["fake_score"])         # 0-1 probability
    """

    def __init__(self):
        self._pipeline  = None
        self._model_id: Optional[str] = None
        self._error:    Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, model_id: str = DEFAULT_MODEL_ID) -> bool:
        """
        Load the model from HuggingFace Hub (or local cache if already downloaded).

        Parameters
        ----------
        model_id : str  HuggingFace model identifier

        Returns
        -------
        bool  True if model loaded successfully, False otherwise.
        """
        if self._pipeline is not None and self._model_id == model_id:
            return True  # Already loaded

        self._pipeline = None
        self._error    = None

        try:
            from transformers import pipeline as hf_pipeline
            from pathlib import Path as _Path

            # Support local fine-tuned model paths as well as HuggingFace IDs
            source = str(_Path(model_id).resolve()) if _Path(model_id).exists() else model_id

            logger.info("Loading model: %s", source)
            self._pipeline = hf_pipeline(
                "image-classification",
                model=source,
                device=-1,          # CPU; set to 0 for first CUDA GPU
                top_k=None,         # return all label probabilities
            )
            self._model_id = model_id   # keep original name/path for display
            logger.info("Model ready.")
            return True

        except ImportError:
            self._error = (
                "PyTorch / Transformers not installed.\n"
                "Fix: pip install torch transformers"
            )
            logger.error(self._error)
            return False

        except Exception as exc:
            self._error = f"Failed to load '{model_id}': {exc}"
            logger.error(self._error)
            return False

    def predict(self, image_path: str) -> dict:
        """
        Run deepfake detection inference on one image.

        Parameters
        ----------
        image_path : str  Absolute or relative path to the image file.

        Returns
        -------
        dict
            fake_score : float  0-1  probability the image is AI/fake
            real_score : float  0-1  probability the image is authentic
            raw        : list   raw HuggingFace output (label + score pairs)
            model_id   : str
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call model_manager.load() first.")

        from PIL import Image
        img     = Image.open(image_path).convert("RGB")
        raw     = self._pipeline(img)           # list of {label, score}

        fake_score, real_score = self._parse_scores(raw)

        return {
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "raw":        raw,
            "model_id":   self._model_id,
        }

    def predict_pil(self, img) -> dict:
        """Same as predict() but accepts a PIL Image directly (for video frames)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        raw = self._pipeline(img.convert("RGB"))
        fake_score, real_score = self._parse_scores(raw)
        return {
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "raw":        raw,
            "model_id":   self._model_id,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._error

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def vit_model(self):
        """Direct access to the underlying ViTForImageClassification — needed for Grad-CAM hooks."""
        return self._pipeline.model if self._pipeline else None

    @property
    def image_processor(self):
        """Direct access to the image processor — needed for Grad-CAM preprocessing."""
        if self._pipeline is None:
            return None
        # HuggingFace pipelines expose this as image_processor or feature_extractor
        return getattr(self._pipeline, "image_processor", None) \
            or getattr(self._pipeline, "feature_extractor", None)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_scores(raw: list) -> tuple[float, float]:
        """
        Convert raw label/score pairs to (fake_score, real_score).

        Handles varying label conventions across different models:
          "Fake" / "Real", "FAKE" / "REAL",
          "AI-generated" / "authentic",  "deepfake" / "genuine", etc.
        """
        fake_score = 0.0
        real_score = 0.0

        _FAKE_WORDS = {"fake", "deepfake", "ai", "generated", "manipulated",
                       "artificial", "synthetic", "forged", "1"}
        _REAL_WORDS = {"real", "authentic", "genuine", "original", "natural",
                       "human", "0"}

        for item in raw:
            label = item["label"].lower().strip()
            score = float(item["score"])

            if any(w in label for w in _FAKE_WORDS):
                fake_score = max(fake_score, score)
            elif any(w in label for w in _REAL_WORDS):
                real_score = max(real_score, score)

        # If only one label matched, infer the other
        if fake_score > 0 and real_score == 0:
            real_score = 1.0 - fake_score
        elif real_score > 0 and fake_score == 0:
            fake_score = 1.0 - real_score
        elif fake_score == 0 and real_score == 0:
            # Fallback: highest score = fake
            if raw:
                fake_score = max(raw, key=lambda x: x["score"])["score"]
                real_score = 1.0 - fake_score

        # Normalise
        total = fake_score + real_score
        if total > 0:
            fake_score /= total
            real_score /= total

        return fake_score, real_score


# ── Module-level singleton ────────────────────────────────────────────────────
# Import and use directly:
#   from detector.model_manager import model_manager
model_manager = ModelManager()
