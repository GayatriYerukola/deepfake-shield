"""
File Utilities
==============
Handles file validation, saving, cleanup, and hashing.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime


# ── Validation constants ──────────────────────────────────────────────────────
MAX_FILE_SIZE_MB   = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/avi"}

ALLOWED_EXTENSIONS = {
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".mp4":  "video",
    ".mov":  "video",
    ".avi":  "video",
}


def validate_file(uploaded_file) -> tuple[bool, str, str]:
    """
    Validate a Streamlit UploadedFile object.

    Returns
    -------
    (is_valid, error_message, file_type)
    file_type is "image" | "video" | ""
    """
    if uploaded_file is None:
        return False, "No file provided.", ""

    # ── Size check ────────────────────────────────────────────────────────
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        return (
            False,
            f"File is {size_mb:.1f} MB — maximum allowed is {MAX_FILE_SIZE_MB} MB.",
            "",
        )

    # ── Extension check ───────────────────────────────────────────────────
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file extension '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            "",
        )

    file_type = ALLOWED_EXTENSIONS[suffix]

    # ── MIME type cross-check ─────────────────────────────────────────────
    mime = (uploaded_file.type or "").lower()
    if file_type == "image" and mime and mime not in ALLOWED_IMAGE_TYPES:
        return False, f"MIME type '{mime}' does not match an image file.", file_type
    if file_type == "video" and mime and mime not in ALLOWED_VIDEO_TYPES:
        return False, f"MIME type '{mime}' does not match a video file.", file_type

    return True, "", file_type


def save_uploaded_file(uploaded_file, directory: str) -> str:
    """
    Write an UploadedFile to disk and return the absolute path.

    Uses a timestamp prefix to avoid collisions.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem   = Path(uploaded_file.name).stem[:40]   # truncate very long names
    suffix = Path(uploaded_file.name).suffix.lower()
    fname  = f"{ts}_{stem}{suffix}"
    dest   = Path(directory) / fname

    uploaded_file.seek(0)
    dest.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)   # reset for any later reads

    return str(dest)


def cleanup_file(file_path: str) -> None:
    """Delete a file from disk, silently ignoring errors."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception:
        pass


def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute a hex digest of the file content for deduplication / audit."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
