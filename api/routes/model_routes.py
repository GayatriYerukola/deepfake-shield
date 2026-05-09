"""
Model Management Routes
=======================
GET  /api/v1/model/status   — check if the neural model is loaded
POST /api/v1/model/load     — download + load the model
"""

from fastapi import APIRouter, HTTPException

from api.models import ModelLoadRequest, ModelStatus

router = APIRouter(tags=["Model"])


@router.get("/model/status", response_model=ModelStatus, summary="Neural model status")
async def model_status():
    from detector.model_manager import model_manager
    return ModelStatus(
        loaded=   model_manager.is_loaded,
        model_id= model_manager.model_id,
        error=    model_manager.load_error,
        message=  "Model ready." if model_manager.is_loaded else
                  "Model not loaded. POST /api/v1/model/load to download.",
    )


@router.post("/model/load", response_model=ModelStatus, summary="Load the neural model")
async def load_model(req: ModelLoadRequest):
    """
    Download and load the specified HuggingFace model.
    First call downloads weights (~85–300 MB); subsequent calls use local cache.
    This endpoint may take 30-120 seconds on first run.
    """
    from detector.model_manager import model_manager
    ok = model_manager.load(req.model_id)
    if not ok:
        raise HTTPException(status_code=500, detail=model_manager.load_error)
    return ModelStatus(
        loaded=   True,
        model_id= model_manager.model_id,
        error=    None,
        message=  f"Model '{model_manager.model_id}' loaded successfully.",
    )
