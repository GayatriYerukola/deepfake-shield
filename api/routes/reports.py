"""
Report Routes
=============
GET /api/v1/reports          — list all scans (paginated)
GET /api/v1/reports/{id}     — get one full report
DELETE /api/v1/reports/{id}  — delete a scan record
"""

from fastapi import APIRouter, HTTPException, Query

from api.database import get_all_scans, get_scan, count_scans
from api.models   import ReportSummary

router = APIRouter(tags=["Reports"])


@router.get("/reports", response_model=list[ReportSummary], summary="List all scan reports")
async def list_reports(
    limit:  int = Query(50,  ge=1, le=500),
    offset: int = Query(0,   ge=0),
):
    """Return a paginated list of scan summaries (newest first)."""
    rows = get_all_scans(limit=limit, offset=offset)
    return [
        ReportSummary(
            scan_id=        r["id"],
            filename=       r["filename"],
            media_type=     r["media_type"],
            verdict=        r.get("verdict"),
            ai_probability= r.get("ai_probability"),
            confidence=     r.get("confidence"),
            generated_at=   r.get("created_at", ""),
        )
        for r in rows
    ]


@router.get("/reports/count", summary="Total number of scans")
async def report_count():
    return {"count": count_scans()}


@router.get("/reports/{scan_id}", summary="Get a full report by scan ID")
async def get_report(scan_id: str):
    """Return the full JSON report for one scan."""
    report = get_scan(scan_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Scan '{scan_id}' not found.")
    return report
