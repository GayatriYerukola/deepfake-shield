"""
Video Deepfake Detector
========================
Extracts frames from a video, analyzes each one, and checks for
temporal inconsistencies that are hallmarks of face-swap deepfakes.

HOW TO UPGRADE
--------------
Replace _analyze_single_frame() with a real model (e.g. a FaceForensics++
trained EfficientNet or TimeSformer for temporal analysis).
The rest of the pipeline — sampling, aggregation, suspicious-frame
collection — stays unchanged.
"""

import cv2
import numpy as np
from pathlib import Path

from .image_detector import (
    _compute_noise_score,
    _compute_color_score,
    _compute_edge_score,
    _compute_frequency_score,
    _detect_faces,
)
from .scoring import compute_final_score


# ── Configuration ─────────────────────────────────────────────────────────────
MAX_SAMPLE_FRAMES   = 12    # Maximum frames to analyze (keeps runtime short)
SUSPICIOUS_THRESHOLD = 0.58  # Frame score above this → listed as suspicious


def analyze_video(file_path: str) -> dict:
    """
    Run the full video analysis pipeline.

    Parameters
    ----------
    file_path : str  Path to the video file on disk.

    Returns
    -------
    dict
        ai_probability      : float | None
        signals             : dict  aggregated signal scores
        suspicious_frames   : list[dict]  frames with high risk scores
        frame_scores        : list[float]
        fps                 : float
        duration            : float  seconds
        total_frames        : int
        technical_details   : dict
        error               : str | None
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return _error_result(
            "Could not open video file. It may be corrupted, password-protected, "
            "or use an unsupported codec."
        )

    fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps if fps > 0 else 0.0

    if total_frames < 1:
        cap.release()
        return _error_result("Video appears to have no readable frames.")

    # ── Sample frame indices evenly across the video ──────────────────────
    n_samples    = min(MAX_SAMPLE_FRAMES, total_frames)
    sample_idxs  = [
        int(i * total_frames / n_samples)
        for i in range(n_samples)
    ]

    frame_scores:      list[float] = []
    frame_signals_all: list[dict]  = []
    suspicious_frames: list[dict]  = []
    faces_found = 0

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp = idx / fps

        score, signals, face_found = _analyze_single_frame(frame)
        frame_scores.append(score)
        frame_signals_all.append(signals)

        if face_found:
            faces_found += 1

        if score >= SUSPICIOUS_THRESHOLD:
            m, s = divmod(int(timestamp), 60)
            suspicious_frames.append({
                "Frame #":    idx,
                "Timestamp":  f"{m:02d}:{s:02d}",
                "Risk Score": f"{score:.0%}",
                "Risk Level": _risk_label(score),
            })

    cap.release()

    if not frame_scores:
        return _error_result("Could not extract any valid frames for analysis.")

    # ── Temporal consistency ──────────────────────────────────────────────
    temporal_score = _temporal_consistency(frame_scores)

    # ── Aggregate per-signal means ────────────────────────────────────────
    agg_signals: dict[str, float] = {}
    all_keys = set().union(*frame_signals_all) if frame_signals_all else set()
    for key in all_keys:
        vals = [s[key] for s in frame_signals_all if key in s]
        if vals:
            agg_signals[key] = float(np.mean(vals))

    agg_signals["Temporal Consistency"] = temporal_score

    ai_probability = compute_final_score(agg_signals)

    return {
        "ai_probability":    ai_probability,
        "signals":           agg_signals,
        "suspicious_frames": suspicious_frames,
        "frame_scores":      frame_scores,
        "fps":               round(fps, 2),
        "duration":          round(duration, 2),
        "total_frames":      total_frames,
        "technical_details": {
            "resolution":     f"{width}×{height}",
            "fps":            round(fps, 2),
            "duration_sec":   round(duration, 2),
            "frames_sampled": len(frame_scores),
            "faces_found_in": f"{faces_found}/{len(frame_scores)} frames",
        },
        "error": None,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _analyze_single_frame(
    frame: np.ndarray,
) -> tuple[float, dict[str, float], bool]:
    """
    Run image-level signals on one video frame.

    Returns (frame_score, signals_dict, face_detected).

    TODO: Replace this with a real temporal or frame-level deepfake model.
    """
    img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    signals = {
        "Noise Pattern":      _compute_noise_score(img_rgb),
        "Color Distribution": _compute_color_score(img_rgb),
        "Edge Consistency":   _compute_edge_score(img_gray),
        "Frequency Artifacts":_compute_frequency_score(img_gray),
    }

    face_detected, face_score = _detect_faces(frame)
    if face_detected:
        signals["Face Anomaly"] = face_score

    frame_score = float(np.mean(list(signals.values())))
    return frame_score, signals, face_detected


def _temporal_consistency(frame_scores: list[float]) -> float:
    """
    Measure temporal consistency of frame scores.

    Face-swap deepfakes often flicker — the score alternates between
    high and low across consecutive frames as the swapped face blends
    inconsistently with lighting changes.

    High variance in frame scores → high temporal inconsistency → more suspicious.

    Returns 0-1 (higher = more suspicious).
    """
    if len(frame_scores) < 2:
        return 0.4

    diffs = [abs(frame_scores[i] - frame_scores[i - 1]) for i in range(1, len(frame_scores))]
    mean_diff = float(np.mean(diffs))
    score_var = float(np.var(frame_scores))

    # High mean diff or high variance = suspicious temporal pattern
    temporal_anomaly = float(np.clip(mean_diff * 3.0 + score_var * 4.0, 0, 1))
    return temporal_anomaly


def _risk_label(score: float) -> str:
    if score >= 0.72:
        return "High"
    elif score >= 0.52:
        return "Medium"
    elif score >= 0.32:
        return "Low"
    return "Minimal"


def _error_result(message: str) -> dict:
    return {
        "ai_probability":    None,
        "signals":           {},
        "suspicious_frames": [],
        "frame_scores":      [],
        "fps":               0.0,
        "duration":          0.0,
        "total_frames":      0,
        "technical_details": {},
        "error":             message,
    }
