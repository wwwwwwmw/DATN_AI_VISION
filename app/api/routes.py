"""
AI Vision API Routes.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.prediction import predict, load_model
from app.core.config import settings

router = APIRouter()


class PredictionResponse(BaseModel):
    predictions: dict
    heatmap_url: Optional[str] = None
    top_class: str
    top_probability: float
    model_version: str
    device: str
    processing_time_ms: float


@router.post("/predict", response_model=PredictionResponse)
async def predict_xray(image: UploadFile = File(...)):
    """
    Upload ảnh X-quang → trả về predictions + Grad-CAM heatmap.
    
    Accepts: JPEG, PNG
    Returns: Predictions for 14 chest pathologies
    """
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Accepted: {', '.join(allowed_types)}"
        )

    # Read file
    contents = await image.read()

    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    try:
        result = predict(contents, generate_heatmap=True)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/model-info")
async def model_info():
    """Thông tin model hiện tại."""
    return {
        "architecture": "DenseNet121",
        "num_classes": settings.NUM_CLASSES,
        "class_names": settings.CLASS_NAMES,
        "class_names_vi": settings.CLASS_NAMES_VI,
        "image_size": settings.IMAGE_SIZE,
        "device": str(settings.DEVICE),
        "model_loaded": settings.MODEL_LOADED,
        "model_path": settings.MODEL_PATH,
    }


@router.post("/warmup")
async def warmup():
    """Pre-load model vào memory (gọi 1 lần khi start service)."""
    try:
        load_model()
        return {"status": "ok", "message": "Model loaded and ready."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")
