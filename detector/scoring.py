"""
Scoring Engine
==============
Aggregates individual detection signals into a final verdict.

Plug-in point: replace compute_final_score() with an ensemble model or
logistic regression trained on real deepfake/authentic pairs.
"""

import numpy as np


# ── Signal weights (heuristic — not trained on data) ─────────────────────────
# Higher weight = signal is considered more reliable.
#
# NOTE: "Neural Model" is handled by the ensemble logic in image_detector.py
# (MODEL_WEIGHT constant) and does NOT flow through compute_final_score().
# The weight below is used only if "Neural Model" is passed directly into
# compute_final_score() from the evaluation script or other callers.
#
# TODO: learn these weights from labeled data (e.g. sklearn LogisticRegression).
_SIGNAL_WEIGHTS: dict[str, float] = {
    # Tier 1 — neural model (when used as a raw signal)
    "Neural Model":          0.70,
    # Tier 2 — classical heuristics
    "Metadata Integrity":    0.09,
    "Noise Pattern":         0.06,
    "Color Distribution":    0.04,
    "Compression Artifacts": 0.04,
    "Edge Consistency":      0.03,
    "Frequency Artifacts":   0.02,
    "Face Anomaly":          0.02,
    # Fallback for any unexpected signal name
    "_default":              0.03,
}

# ── Verdict thresholds ────────────────────────────────────────────────────────
_THRESHOLDS = {
    "Likely Real":            (0.00, 0.32),
    "Inconclusive":           (0.32, 0.52),
    "Suspicious":             (0.52, 0.72),
    "Likely AI / Manipulated":(0.72, 1.01),
}


def compute_final_score(signals: dict[str, float]) -> float:
    """
    Weighted average of detection signals, with small calibration noise.

    Parameters
    ----------
    signals : dict  signal_name -> score (0-1, higher = more suspicious)

    Returns
    -------
    float  final AI probability 0-1
    """
    if not signals:
        return 0.5

    weighted_sum  = 0.0
    total_weight  = 0.0

    for name, score in signals.items():
        w = _SIGNAL_WEIGHTS.get(name, _SIGNAL_WEIGHTS["_default"])
        weighted_sum += score * w
        total_weight += w

    base = weighted_sum / total_weight if total_weight else 0.5

    # Small reproducible jitter to acknowledge model uncertainty.
    # Remove this when you plug in a real calibrated model.
    rng   = np.random.default_rng(seed=int(base * 1e6))
    jitter = rng.uniform(-0.04, 0.04)

    return float(np.clip(base + jitter, 0.0, 1.0))


def compute_confidence(signals: dict[str, float]) -> float:
    """
    Confidence reflects how much the signals agree with each other.
    If all signals point the same direction: high confidence.
    If signals are contradictory: low confidence.
    """
    if len(signals) < 2:
        return 0.40

    values = list(signals.values())
    std    = float(np.std(values))

    # std=0 → perfect agreement → confidence 0.95
    # std=0.5 → maximum disagreement → confidence ~0.35
    confidence = max(0.30, 0.95 - std * 1.2)
    return round(confidence, 2)


def compute_verdict(ai_probability: float) -> str:
    """Map a probability score to a human-readable risk verdict."""
    for label, (lo, hi) in _THRESHOLDS.items():
        if lo <= ai_probability < hi:
            return label
    return "Inconclusive"


def build_explanation(verdict: str, signals: dict[str, float], face_detected: bool) -> str:
    """
    Generate a plain-English explanation of the result.
    """
    top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    top_name    = top_signals[0][0] if top_signals else "multiple signals"

    face_note = " Facial features were detected, which are common targets for deepfake manipulation." \
        if face_detected else ""

    explanations = {
        "Likely Real": (
            f"The analysis found no strong indicators of AI generation or manipulation. "
            f"The strongest contributing signal was '{top_name}'.{face_note} "
            f"This result suggests the media is likely authentic, though statistical analysis "
            f"cannot provide absolute certainty."
        ),
        "Inconclusive": (
            f"The signals are mixed and do not point clearly in either direction. "
            f"'{top_name}' was the most prominent signal.{face_note} "
            f"We recommend manual review or analysis with a specialized forensic tool."
        ),
        "Suspicious": (
            f"Several signals are consistent with AI generation or digital manipulation. "
            f"The primary concern is '{top_name}'.{face_note} "
            f"This does not confirm the media is fake, but warrants further investigation."
        ),
        "Likely AI / Manipulated": (
            f"Multiple strong signals suggest this media may be AI-generated or digitally manipulated. "
            f"The most significant indicator is '{top_name}'.{face_note} "
            f"We recommend treating this media with caution until verified by other means."
        ),
    }

    return explanations.get(verdict, "Analysis complete. Review the signal breakdown for details.")


def build_suspicious_signals(signals: dict[str, float], threshold: float = 0.52) -> list[str]:
    """
    Return human-readable descriptions of signals that exceeded the threshold.
    """
    descriptions = {
        "Metadata Integrity":    "Missing or incomplete camera metadata (common in AI-generated images)",
        "Noise Pattern":         "Unusual noise patterns inconsistent with typical camera sensors",
        "Color Distribution":    "Color histogram anomalies detected in the image",
        "Compression Artifacts": "Irregular JPEG compression patterns (ELA analysis)",
        "Edge Consistency":      "Inconsistent edge sharpness across the image",
        "Frequency Artifacts":   "Frequency-domain artifacts suggesting GAN upsampling operations",
        "Face Anomaly":          "Facial region shows indicators associated with deepfake manipulation",
    }

    flagged = []
    for name, score in signals.items():
        if score >= threshold:
            desc = descriptions.get(name, f"{name} scored {score:.0%}")
            flagged.append(f"{desc} ({score:.0%} risk)")

    return flagged
