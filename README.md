# 🧠 AI Vision — DenseNet121 Chest X-ray Classification

> **Port:** 8001 | **Framework:** FastAPI + PyTorch

## Quick Start

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Tiền xử lý dữ liệu
python scripts/preprocess_chexpert.py
python scripts/preprocess_nih.py
python scripts/merge_datasets.py

# Huấn luyện model
python training/train.py --config training/config.yaml

# Đánh giá model
python training/evaluate.py --checkpoint weights/best_densenet121.pth

# Chạy API server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Tài liệu chi tiết

Xem [docs/ai-vision/README.md](../docs/ai-vision/README.md) cho checklist nghiệp vụ đầy đủ.

## Cấu trúc

```
ai-vision/
├── app/                 # FastAPI application
│   ├── api/             # API routes & schemas
│   ├── core/            # Config & dependencies
│   └── services/        # Prediction & Grad-CAM
├── model/               # Kiến trúc mạng, losses, metrics
├── data/                # Dataset classes & transforms
├── training/            # Train & evaluate scripts
├── notebooks/           # Jupyter EDA notebooks
├── weights/             # Trained model weights (.pth)
└── scripts/             # Data preprocessing scripts
```

## Biến môi trường

```env
MODEL_PATH=weights/best_densenet121.pth
DEVICE=cuda
BATCH_SIZE=32
CONFIDENCE_THRESHOLD=0.5
```
# DATN_AI_VISION
