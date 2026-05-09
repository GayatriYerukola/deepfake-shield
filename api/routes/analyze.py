"""
Analysis Routes
===============
POST /api/v1/analyze/image   — upload + analyze an image
POST /api/v1/analyze/video   — upload + analyze a video
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.database  import insert_scan, next_scan_id
from api.models    import AnalysisResponse
from utils.file_utils  import validate_file
from utils.report_utils import generate_report, save_report

router = APIRouter(tags=["Analysis"])

# Allowed extensions by media type
_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
_VIDEO_EXT = {".mp4", ".mov", ".avi"}


@router.post("/analyze/image", response_model=AnalysisResponse, summary="Analyze an image")
async def analyze_image_endpoint(
    file:      UploadFile = File(..., description="JPG, JPEG, or PNG image"),
    use_model: bool       = Form(True,  description="Include neural model in analysis"),
):
    """
    Upload an image and receive a full deepfake risk report.

    - **file**: the image to analyze (JPG/JPEG/PNG, max 200 MB)
    - **use_model**: set false to use heuristics only (faster, less accurate)
    """
    return await _run_analysis(file, "image", use_model)


@router.post("/analyze/video", response_model=AnalysisResponse, summary="Analyze a video")
async def analyze_video_endpoint(
    file: UploadFile = File(..., description="MP4, MOV, or AVI video"),
):
    """
    Upload a video and receive a deepfake risk report with per-frame analysis.

    - **file**: the video to analyze (MP4/MOV/AVI, max 200 MB)
    """
    return await _run_analysis(file, "video", use_model=False)


# ── Shared logic ──────────────────────────────────────────────────────────────

async def _run_analysis(
    upload: UploadFile,
    expected_type: str,
    use_model: bool,
) -> AnalysisResponse:
    # ── Validate ──────────────────────────────────────────────────────────
    is_valid, err_msg, file_type = validate_file(upload)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)
    if file_type != expected_type:
        raise HTTPException(
            status_code=415,
            detail=f"Expected {expected_type} file, got {file_type}."
        )

    # ── Save to temp file ─────────────────────────────────────────────────
    suffix = Path(upload.filename or "upload").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        tmp_path = tmp.name

    try:
        # ── Run detector ──────────────────────────────────────────────────
        if file_type == "image":
            from detector.image_detector import analyze_image
            result = analyze_image(tmp_path, use_model=use_model)
        else:
            from detector.video_detector import analyze_video
            result = analyze_video(tmp_path)

        if result.get("error"):
            raise HTTPException(status_code=422, detail=result["error"])

        # ── Build report ──────────────────────────────────────────────────
        report   = generate_report(
            filename=upload.filename or "upload",
            file_type=file_type,
            result=result,
        )
        scan_id  = next_scan_id()
        report["scan_id"] = scan_id
        save_report(report)
        insert_scan(scan_id, report)

        td = report.get("technical_details", {})

        return AnalysisResponse(
            scan_id=scan_id,
            status="success",
            filename=report["filename"],
            media_type=report["media_type"],
            verdict=report.get("verdict"),
            ai_probability=report.get("ai_probability"),
            confidence_level=report.get("confidence_level"),
            explanation=report.get("explanation"),
            suspicious_signals=report.get("suspicious_signals", []),
            signal_breakdown=report.get("signal_breakdown", {}),
            suspicious_frames=report.get("suspicious_frames", []),
            model_used=bool(td.get("model_used", False)),
            model_id=td.get("model_id"),
            technical_details=td,
            generated_at=report.get("generated_at", ""),
            disclaimer=report.get("disclaimer", ""),
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)
