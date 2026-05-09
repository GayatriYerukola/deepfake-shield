"""
DeepFake Shield — FastAPI Backend
==================================
A production-ready REST API that exposes the same detection pipeline
used by the Streamlit frontend — so any client (mobile, browser extension,
CLI tool, research script) can call it programmatically.

Run
---
  uvicorn api.main:app --reload --port 8000

Interactive docs (auto-generated):
  http://localhost:8000/docs      ← Swagger UI
  http://localhost:8000/redoc     ← ReDoc

Example curl
------------
  curl -X POST http://localhost:8000/api/v1/analyze/image \
       -F "file=@photo.jpg" \
       -F "use_model=true"
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.database import init_db
from api.models   import HealthResponse
from api.routes   import analyze, reports, model_routes


# ── Lifespan: runs on startup and shutdown ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Path("uploads").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    init_db()
    yield
    # Shutdown (nothing to clean up for now)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepFake Shield API",
    description=(
        "REST API for AI-generated and deepfake media detection.\n\n"
        "**Disclaimer:** Results are probabilistic risk estimates only — "
        "not forensic proof. Do not use for legal decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

PREFIX = "/api/v1"

app.include_router(analyze.router,       prefix=PREFIX)
app.include_router(reports.router,       prefix=PREFIX)
app.include_router(model_routes.router,  prefix=PREFIX)


# ── Root and health ───────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "DeepFake Shield API", "docs": "/docs"}


@app.get(f"{PREFIX}/health", response_model=HealthResponse, tags=["Health"])
async def health():
    from detector.model_manager import model_manager
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_ready=model_manager.is_loaded,
    )


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
