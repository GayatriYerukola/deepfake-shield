"""
Image Deepfake Detector
========================
Two-tier pipeline:

  Tier 1 — Neural model (primary signal, high weight)
    Uses the HuggingFace model loaded by model_manager.
    Skipped gracefully if the model is not loaded.

  Tier 2 — Classical heuristics (supporting signals)
    Metadata, noise, colour, ELA, edge, FFT analysis.
    Always runs as a fallback and as supplementary evidence.

Ensemble formula (when model is available):
    final_score = MODEL_WEIGHT * model_score
                + (1 - MODEL_WEIGHT) * heuristic_score

To swap the neural model, change DEFAULT_MODEL_ID in model_manager.py.
"""

import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from .metadata_checker import check_metadata
from .model_manager    import model_manager
from .scoring          import compute_final_score

# Weight given to the neural model vs the heuristics ensemble.
# Raise toward 1.0 as you gain confidence in your chosen model.
MODEL_WEIGHT = 0.70


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_image(file_path: str, use_model: bool = True) -> dict:
    """
    Full image analysis pipeline.

    Parameters
    ----------
    file_path  : str   Path to image file on disk.
    use_model  : bool  Whether to include the neural model signal.
                       Set False for heuristics-only evaluation.

    Returns
    -------
    dict
        ai_probability   : float | None   0-1, higher = more likely fake/AI
        signals          : dict           signal_name -> score 0-1
        technical_details: dict
        face_detected    : bool
        model_used       : bool
        error            : str | None
    """
    try:
        img_pil = Image.open(file_path)
        img_cv  = cv2.imread(file_path)

        if img_cv is None:
            return _error_result("Could not read image — file may be corrupted or unsupported.")

        img_pil  = img_pil.convert("RGB")
        img_rgb  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        h, w = img_gray.shape
        if w < 16 or h < 16:
            return _error_result("Image too small for analysis (minimum 16×16 px).")

        # ── Tier 2: Classical heuristics (always run) ─────────────────────
        meta_result    = check_metadata(file_path)
        metadata_score = meta_result["suspicion_score"]

        heuristic_signals = {
            "Metadata Integrity":    metadata_score,
            "Noise Pattern":         _compute_noise_score(img_rgb),
            "Color Distribution":    _compute_color_score(img_rgb),
            "Compression Artifacts": _compute_ela_score(file_path, img_pil),
            "Edge Consistency":      _compute_edge_score(img_gray),
            "Frequency Artifacts":   _compute_frequency_score(img_gray),
        }

        face_detected, face_score = _detect_faces(img_cv)
        if face_detected:
            heuristic_signals["Face Anomaly"] = face_score

        heuristic_score = compute_final_score(heuristic_signals)

        # ── Tier 1: Neural model (primary signal when loaded) ─────────────
        model_used  = False
        model_score = None
        model_id    = None

        if use_model and model_manager.is_loaded:
            try:
                pred        = model_manager.predict(file_path)
                model_score = pred["fake_score"]
                model_id    = pred["model_id"]
                model_used  = True
            except Exception:
                pass    # Graceful fallback to heuristics-only

        # ── Ensemble ──────────────────────────────────────────────────────
        all_signals = dict(heuristic_signals)   # copy

        if model_used and model_score is not None:
            all_signals["Neural Model"] = model_score
            ai_probability = (
                MODEL_WEIGHT       * model_score
                + (1 - MODEL_WEIGHT) * heuristic_score
            )
        else:
            ai_probability = heuristic_score

        return {
            "ai_probability":    float(ai_probability),
            "signals":           all_signals,
            "technical_details": {
                "image_size":    f"{w}×{h}",
                "channels":      img_rgb.shape[2] if img_rgb.ndim == 3 else 1,
                "format":        Path(file_path).suffix.upper().lstrip("."),
                "face_detected": face_detected,
                "model_used":    model_used,
                "model_id":      model_id,
                "metadata":      meta_result,
            },
            "face_detected": face_detected,
            "model_used":    model_used,
            "error":         None,
        }

    except FileNotFoundError:
        return _error_result(f"File not found: {file_path}")
    except Exception as exc:
        return _error_result(f"Analysis failed: {exc}")


# ── Classical signal functions ────────────────────────────────────────────────
# Each returns float 0-1 (higher = more suspicious).
# Replace any body with real model inference — the rest of the pipeline is unchanged.

def _compute_noise_score(img_rgb: np.ndarray) -> float:
    """
    Uniform, low-variance noise = suspicious (AI images are "too clean").
    Camera sensors produce spatially varied photon/read noise.
    """
    blurred   = cv2.GaussianBlur(img_rgb, (5, 5), 0).astype(np.float64)
    noise     = img_rgb.astype(np.float64) - blurred
    noise_std = float(np.std(noise))

    block = 32
    h, w  = img_rgb.shape[:2]
    local_stds = [
        np.std(noise[y:y+block, x:x+block])
        for y in range(0, h-block, block)
        for x in range(0, w-block, block)
    ]
    if not local_stds:
        return 0.5

    cv = float(np.std(local_stds)) / (float(np.mean(local_stds)) + 1e-8)
    uniformity = float(np.clip(1.0 - cv * 2.0, 0, 1))
    low_noise  = float(np.clip(1.0 - noise_std / 12.0, 0, 1))
    return float(np.clip(0.55 * uniformity + 0.45 * low_noise, 0, 1))


def _compute_color_score(img_rgb: np.ndarray) -> float:
    """Unnaturally smooth per-channel histograms suggest AI synthesis."""
    scores = []
    for ch in range(3):
        hist = cv2.calcHist([img_rgb], [ch], None, [256], [0, 256]).flatten()
        hist /= hist.sum() + 1e-8
        smoothness = 1.0 / (float(np.std(np.diff(hist))) * 100 + 1)
        scores.append(smoothness)
    return float(np.clip(float(np.mean(scores)) * 2.2, 0, 1))


def _compute_ela_score(file_path: str, img_pil: Image.Image) -> float:
    """
    Error Level Analysis: manipulated/generated regions compress differently
    from the surrounding image at a known quality level.
    Less reliable for PNG (down-weighted automatically).
    """
    try:
        buf = io.BytesIO()
        img_pil.save(buf, "JPEG", quality=75)
        buf.seek(0)
        recomp   = Image.open(buf).convert("RGB")
        diff     = np.abs(
            np.array(img_pil, np.float64) - np.array(recomp, np.float64)
        )
        ela_mean = float(diff.mean())
        norm     = float(np.clip(ela_mean / 30.0, 0, 1))
        return norm * (0.5 if Path(file_path).suffix.lower() == ".png" else 1.0)
    except Exception:
        return 0.5


def _compute_edge_score(img_gray: np.ndarray) -> float:
    """Over-smooth or too-noisy edge density deviates from natural photo range."""
    n         = float(img_gray.size)
    dens_lo   = cv2.Canny(img_gray,  30, 100).sum() / (255 * n)
    dens_hi   = cv2.Canny(img_gray, 100, 200).sum() / (255 * n)

    if dens_lo < 0.02:
        smoothness = 0.80
    elif dens_lo > 0.55:
        smoothness = 0.65
    else:
        smoothness = 0.20

    ratio_score = float(np.clip(abs(dens_hi / (dens_lo + 1e-8) - 0.45) * 0.8, 0, 1))
    return float(np.clip(0.6 * smoothness + 0.4 * ratio_score, 0, 1))


def _compute_frequency_score(img_gray: np.ndarray) -> float:
    """
    GAN transposed-convolution upsampling leaves periodic grid artifacts
    visible as variance spikes in the FFT mid-frequency band.
    See: Frank et al., 2020 — Leveraging Frequency Analysis for Deepfake Detection.
    """
    f   = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float64)))
    mag = np.log1p(np.abs(f))
    mag /= mag.max() + 1e-8
    h, w = mag.shape
    mid  = mag[h//4: 3*h//4, w//4: 3*w//4]
    return float(np.clip(float(np.var(mid)) * 6.0, 0, 1))


def _detect_faces(img_cv: np.ndarray) -> tuple[bool, float]:
    """
    Haar-cascade face detection. Faces raise deepfake risk category;
    score is a conservative placeholder (0.40).
    TODO: replace with a face-specific deepfake model (e.g. RetinaFace + EfficientNet).
    """
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return False, 0.0
        return True, 0.40
    except Exception:
        return False, 0.0


def _error_result(message: str) -> dict:
    return {
        "ai_probability":    None,
        "signals":           {},
        "technical_details": {},
        "face_detected":     False,
        "model_used":        False,
        "error":             message,
    }
