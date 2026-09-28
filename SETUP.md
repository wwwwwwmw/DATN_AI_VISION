# 🧠 AI Vision — Hướng dẫn Chạy

## Yêu cầu
- Python 3.11+
- PyTorch 2.x (GPU hoặc CPU)
- ~2GB disk cho pretrained weights

## Quick Start

```cmd
REM 1. Tạo virtual environment
cd ai-vision
python -m venv venv
venv\Scripts\activate

REM 2. Cài PyTorch (chọn 1 trong 2):
REM --- Có GPU NVIDIA (CUDA 12.4) ---
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

REM --- Không có GPU (CPU only) ---
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM 3. Cài dependencies còn lại
pip install -r requirements.txt

REM 4. Chạy API server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Huấn luyện Model

```cmd
REM Huấn luyện trên CheXpert dataset
python training/train.py --data-dir ../../CheXpert --epochs 50 --batch-size 32

REM Kết quả lưu tại:
REM   weights/best_densenet121.pth      ← Model tốt nhất
REM   weights/training_history.json     ← Lịch sử huấn luyện
```

### Chạy trên máy KHÔNG có GPU

File `best_densenet121.pth` được lưu với `map_location` linh hoạt.
Copy file `.pth` sang máy khác → chạy inference trên CPU bình thường:

```python
from model.densenet121 import ChestXrayDenseNet121
model = ChestXrayDenseNet121.load_from_checkpoint(
    'weights/best_densenet121.pth',
    device='cpu'  # ← chạy CPU
)
```

## Test API

```cmd
REM Health check
curl http://localhost:8001/health

REM Predict (upload ảnh X-quang)
curl -X POST http://localhost:8001/api/predict ^
  -F "image=@test_xray.jpg"

REM Model info
curl http://localhost:8001/api/model-info

REM Warmup (pre-load model)
curl -X POST http://localhost:8001/api/warmup
```

## Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| POST | /api/predict | Upload ảnh → predictions + heatmap |
| GET | /api/model-info | Thông tin model |
| POST | /api/warmup | Pre-load model |
| GET | /health | Health check |
| GET | /heatmaps/{file} | Xem heatmap |
