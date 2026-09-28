"""
AI Vision — Configuration.
"""
import os
import torch
from pathlib import Path


class Settings:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH = os.environ.get('MODEL_PATH', str(BASE_DIR / 'weights' / 'best_densenet121.pth'))
    HEATMAP_DIR = str(BASE_DIR / 'outputs' / 'heatmaps')

    # Model
    NUM_CLASSES = 14
    IMAGE_SIZE = 224
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.5'))

    # Device
    DEVICE_STR = os.environ.get('DEVICE', 'auto')
    if DEVICE_STR == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(DEVICE_STR)

    # Status
    MODEL_LOADED = False

    # Class names (14 CheXpert classes)
    CLASS_NAMES = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices',
    ]

    # Vietnamese names
    CLASS_NAMES_VI = {
        'No Finding': 'Không phát hiện bất thường',
        'Enlarged Cardiomediastinum': 'Phì đại trung thất',
        'Cardiomegaly': 'Tim to',
        'Lung Opacity': 'Mờ phổi',
        'Lung Lesion': 'Tổn thương phổi',
        'Edema': 'Phù phổi',
        'Consolidation': 'Đông đặc phổi',
        'Pneumonia': 'Viêm phổi',
        'Atelectasis': 'Xẹp phổi',
        'Pneumothorax': 'Tràn khí màng phổi',
        'Pleural Effusion': 'Tràn dịch màng phổi',
        'Pleural Other': 'Bất thường màng phổi khác',
        'Fracture': 'Gãy xương',
        'Support Devices': 'Thiết bị hỗ trợ',
    }


settings = Settings()
