"""
Metadata Checker
================
Inspects EXIF and file metadata for AI-generation signals.
This is one of the most reliable heuristics available without a trained model —
many AI tools either strip metadata or leave recognizable fingerprints.
"""

import os
from pathlib import Path
from PIL import Image, ExifTags
from datetime import datetime


# Known AI/editing software signatures found in the "Software" EXIF field
_AI_SOFTWARE_MARKERS = [
    "midjourney", "stable diffusion", "dall-e", "dall·e", "firefly",
    "generative", "ai generated", "imagen", "craiyon", "nightcafe",
    "artbreeder", "runway", "pika", "kling", "sora", "leonardo",
    "adobe firefly", "canva ai", "bing image creator",
]

# Editing software that doesn't necessarily mean fake, but flags as "processed"
_EDITING_SOFTWARE = [
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "darktable", "rawtherapee", "luminar",
]


def check_metadata(file_path: str) -> dict:
    """
    Analyze image metadata for AI-generation and manipulation indicators.

    Returns:
        suspicion_score : float  0-1, higher = more suspicious
        has_exif        : bool
        camera_info     : dict | None
        ai_markers      : list[str]   strings that triggered AI detection
        software_used   : str | None
        flags           : list[str]   human-readable flags for the report
    """
    result = {
        "suspicion_score": 0.5,
        "has_exif": False,
        "camera_info": None,
        "ai_markers": [],
        "software_used": None,
        "flags": [],
    }

    try:
        img = Image.open(file_path)
        exif_raw = img._getexif()  # Returns None for PNG/non-JPEG

        # ── No EXIF at all ────────────────────────────────────────────────
        if exif_raw is None:
            result["flags"].append("No EXIF metadata found (common in AI-generated images)")
            # PNG files legitimately lack EXIF, so penalize less
            ext = Path(file_path).suffix.lower()
            result["suspicion_score"] = 0.55 if ext == ".png" else 0.65
            return result

        # ── Decode EXIF tags ───────────────────────────────────────────────
        result["has_exif"] = True
        exif = {
            ExifTags.TAGS.get(k, k): v
            for k, v in exif_raw.items()
        }

        # Camera make / model
        make  = exif.get("Make", "")
        model = exif.get("Model", "")
        software = str(exif.get("Software", "")).strip()

        if make or model:
            result["camera_info"] = {"make": make, "model": model}

        if software:
            result["software_used"] = software

        # ── Check for AI tool fingerprints ────────────────────────────────
        software_lower = software.lower()

        for marker in _AI_SOFTWARE_MARKERS:
            if marker in software_lower:
                result["ai_markers"].append(f'Software field contains "{marker}"')

        # Check ImageDescription and UserComment for AI markers
        for field in ("ImageDescription", "UserComment", "XPComment"):
            val = str(exif.get(field, "")).lower()
            for marker in _AI_SOFTWARE_MARKERS:
                if marker in val:
                    result["ai_markers"].append(f'{field} contains "{marker}"')

        # ── Score computation ─────────────────────────────────────────────
        score = 0.2  # Start optimistic: has EXIF with some content

        if result["ai_markers"]:
            # Strong signal — AI tool explicitly named
            score = 0.95
            result["flags"].append(
                f"AI generation software detected in metadata: {result['ai_markers'][0]}"
            )
        elif not make and not model:
            # Has EXIF but no camera info — suspicious
            score = 0.60
            result["flags"].append("EXIF present but missing camera make/model")
        else:
            score = 0.20
            result["flags"].append(f"Camera metadata found: {make} {model}".strip())

        # Editing software raises suspicion moderately
        for ed in _EDITING_SOFTWARE:
            if ed in software_lower:
                score = min(score + 0.15, 0.85)
                result["flags"].append(f"Image editing software detected: {software}")
                break

        result["suspicion_score"] = float(score)

    except Exception as e:
        result["flags"].append(f"Could not parse metadata: {str(e)}")
        result["suspicion_score"] = 0.50

    return result
