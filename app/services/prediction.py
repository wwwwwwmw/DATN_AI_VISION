"""
Prediction Service — Inference pipeline.
Load model → preprocess → predict → Grad-CAM → return results.
"""
import torch
import numpy as np
import cv2
import logging
import time
from pathlib import Path
from PIL import Image
from torchvision import transforms

from app.core.config import settings
from model.densenet121 import ChestXrayDenseNet121
from app.services.gradcam import GradCAM

logger = logging.getLogger(__name__)

# Global model instance
_model = None
_gradcam = None
_transform = None


def get_transform():
    """Preprocessing pipeline cho inference."""
    global _transform
    if _transform is None:
        _transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            ),
        ])
    return _transform


def load_model():
    """Load model from checkpoint (lazy loading)."""
    global _model, _gradcam

    if _model is not None:
        return _model

    model_path = settings.MODEL_PATH

    if Path(model_path).exists():
        logger.info(f"Loading model from {model_path} on {settings.DEVICE}")
        _model = ChestXrayDenseNet121.load_from_checkpoint(
            model_path,
            num_classes=settings.NUM_CLASSES,
            device=str(settings.DEVICE),
        )
        _gradcam = GradCAM(_model)
        settings.MODEL_LOADED = True
        logger.info("Model loaded successfully!")
    else:
        logger.warning(f"Model file not found at {model_path}. Using pretrained ImageNet weights.")
        _model = ChestXrayDenseNet121(
            num_classes=settings.NUM_CLASSES,
            pretrained=True,
        )
        _model.to(settings.DEVICE)
        _model.eval()
        _gradcam = GradCAM(_model)
        settings.MODEL_LOADED = True  # Still usable (untrained)

    return _model


def predict(image_bytes: bytes, generate_heatmap: bool = True):
    """
    Run prediction on an X-ray image.

    Args:
        image_bytes: Raw image bytes
        generate_heatmap: Whether to generate Grad-CAM heatmap

    Returns:
        dict with predictions, heatmap_path, processing_time
    """
    start_time = time.time()

    # Load model
    model = load_model()

    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Cannot decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    # Preprocess
    transform = get_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(settings.DEVICE)

    # Predict
    probs = model.predict(input_tensor)
    probs = probs[0].cpu().numpy()

    # Build predictions dict
    predictions = {}
    for i, class_name in enumerate(settings.CLASS_NAMES):
        predictions[class_name] = float(round(probs[i], 4))

    # Sort by probability
    sorted_predictions = dict(
        sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    )

    # Generate Grad-CAM heatmap for top finding
    heatmap_path = None
    if generate_heatmap and _gradcam is not None:
        top_class_idx = int(np.argmax(probs))
        try:
            timestamp = int(time.time() * 1000)
            heatmap_filename = f"heatmap_{timestamp}_{settings.CLASS_NAMES[top_class_idx].replace(' ', '_')}.jpg"
            heatmap_full_path = Path(settings.HEATMAP_DIR) / heatmap_filename

            _gradcam.save_heatmap(
                input_tensor, img_bgr, top_class_idx, str(heatmap_full_path)
            )
            heatmap_path = f"/heatmaps/{heatmap_filename}"
        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")

    processing_time = (time.time() - start_time) * 1000

    return {
        'predictions': sorted_predictions,
        'heatmap_url': heatmap_path,
        'top_class': settings.CLASS_NAMES[int(np.argmax(probs))],
        'top_probability': float(round(np.max(probs), 4)),
        'model_version': 'densenet121_v1.0',
        'device': str(settings.DEVICE),
        'processing_time_ms': round(processing_time, 2),
    }
