"""
Pydantic request / response models for the FastAPI layer.
These define the exact JSON shapes the API sends and receives.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class SignalBreakdown(BaseModel):
    name:  str
    score: float = Field(..., ge=0, le=1)


# ── Analysis response ────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    scan_id:           str
    status:            str                        # "success" | "error"
    filename:          str
    media_type:        str                        # "image" | "video"
    verdict:           Optional[str]   = None
    ai_probability:    Optional[float] = Field(None, ge=0, le=1)
    confidence_level:  Optional[float] = Field(None, ge=0, le=1)
    explanation:       Optional[str]   = None
    suspicious_signals: list[str]      = []
    signal_breakdown:  dict[str, float] = {}
    suspicious_frames: list[dict]      = []
    model_used:        bool            = False
    model_id:          Optional[str]   = None
    technical_details: dict[str, Any]  = {}
    generated_at:      str             = ""
    error_message:     Optional[str]   = None
    disclaimer:        str             = (
        "This is a probabilistic risk estimate only. "
        "Not forensic evidence. Do not use for legal decisions."
    )


# ── Report summary (for list endpoint) ───────────────────────────────────────

class ReportSummary(BaseModel):
    scan_id:        str
    filename:       str
    media_type:     str
    verdict:        Optional[str]
    ai_probability: Optional[float]
    confidence:     Optional[float]
    generated_at:   str


# ── Model status ──────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    loaded:     bool
    model_id:   Optional[str]
    error:      Optional[str]
    message:    str


class ModelLoadRequest(BaseModel):
    model_id: str = "dima806/deepfake_vs_real_image_detection"


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str   = "ok"
    version:     str   = "1.0.0"
    model_ready: bool  = False
