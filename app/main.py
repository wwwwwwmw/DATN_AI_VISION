"""
AI Vision Service - FastAPI Application
DenseNet121 Chest X-ray Classification + Grad-CAM + Training Dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import os

from app.core.config import settings
from app.api.routes import router as api_router
from app.api.training_routes import router as training_router

app = FastAPI(
    title="MedTech AI Vision",
    description="DenseNet121 Chest X-ray Classification API with Grad-CAM heatmaps",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for heatmaps
heatmap_dir = Path("outputs/heatmaps")
heatmap_dir.mkdir(parents=True, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=str(heatmap_dir)), name="heatmaps")

# API Routes
app.include_router(api_router, prefix="/api")
app.include_router(training_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the landing page with link to dashboard."""
    return """<!DOCTYPE html>
    <html><head><meta charset="UTF-8"><title>MedTech AI Vision</title>
    <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0b0f1a;font-family:'Segoe UI',system-ui,sans-serif;color:#e0e6f0}
    .card{text-align:center;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:48px 56px;backdrop-filter:blur(12px)}
    .pulse{display:inline-block;width:12px;height:12px;background:#22c55e;border-radius:50%;margin-right:8px;animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,0.5)}50%{box-shadow:0 0 0 10px rgba(34,197,94,0)}}
    h1{font-size:28px;margin-bottom:12px;background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .status{font-size:18px;color:#22c55e;margin-bottom:24px}
    .info{color:#8892a8;font-size:14px;line-height:2}
    a{color:#60a5fa;text-decoration:none}a:hover{text-decoration:underline}
    </style></head><body>
    <div class="card">
        <h1>MedTech AI Vision Service</h1>
        <div class="status"><span class="pulse"></span> Service dang hoat dong</div>
        <div class="info">
            Model: DenseNet121 (14 classes)<br>
            API: <a href="/api/predict">/api/predict</a> | <a href="/docs">/docs (Swagger)</a><br>
            Health: <a href="/health">/health</a><br>
            <br>
            <a href="/dashboard" style="font-size:16px;font-weight:600;color:#a78bfa">Training Dashboard</a>
        </div>
    </div></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve training dashboard."""
    template_path = Path(__file__).parent.parent / "templates" / "dashboard.html"
    if template_path.exists():
        return template_path.read_text(encoding='utf-8')
    return "<h1>Dashboard template not found</h1>"


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-vision", "model_loaded": settings.MODEL_LOADED}