"""
AI Vision - Training Script for CheXpert X-ray Classification.

Usage:
    python -m app.train --data-dir ./data/chexpert --epochs 20 --batch-size 32
    python -m app.train --data-dir ./data/chexpert --train-csv ./data/train.csv --epochs 20

Co the chay tren CPU hoac GPU (tu dong detect).
Ket qua luu vao: weights/best_densenet121.pth
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.densenet121 import ChestXrayDenseNet121
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class CheXpertDataset(Dataset):
    """CheXpert CSV format dataset. Labels: 0=neg, 1=pos, -1=uncertain, blank=unknown."""

    def __init__(self, csv_path, data_dir, transform=None, policy='ones'):
        import csv
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_names = settings.CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.samples = []

        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_file}")
            return

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.data_dir / row['Path']
                if not img_path.exists():
                    continue
                labels = []
                for cls in self.class_names:
                    val = row.get(cls, '')
                    if val == '' or val == 'nan':
                        labels.append(0.0)
                    elif float(val) == -1:
                        labels.append(1.0 if policy == 'ones' else 0.0)
                    else:
                        labels.append(max(0.0, float(val)))
                self.samples.append({'path': str(img_path), 'labels': labels})

        logger.info(f"Loaded {len(self.samples)} samples from {csv_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(sample['labels'])


class SimpleImageFolderDataset(Dataset):
    """Fallback: data_dir/ClassName/images.jpg (one subfolder per class)."""

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.class_names = settings.CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.samples = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = Path(data_dir) / class_name.replace(' ', '_')
            if not class_dir.exists():
                continue
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in class_dir.glob(ext):
                    labels = [0.0] * self.num_classes
                    labels[class_idx] = 1.0
                    self.samples.append({'path': str(img_path), 'labels': labels})

        logger.info(f"Loaded {len(self.samples)} images from folder structure")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(sample['labels'])


def get_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def compute_auc(labels, predictions):
    try:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) < 2:
                continue
            aucs.append(roc_auc_score(labels[:, i], predictions[:, i]))
        return np.mean(aucs) if aucs else 0.0
    except ImportError:
        return 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    return total_loss / max(n, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            n += 1
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
    avg_loss = total_loss / max(n, 1)
    if all_labels:
        mean_auc = compute_auc(np.concatenate(all_labels), np.concatenate(all_preds))
    else:
        mean_auc = 0.0
    return avg_loss, mean_auc


def main():
    parser = argparse.ArgumentParser(description='Train DenseNet121 for Chest X-ray Classification')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--train-csv', type=str, default=None, help='Path to train.csv (CheXpert)')
    parser.add_argument('--val-csv', type=str, default=None, help='Path to valid.csv (CheXpert)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--policy', type=str, default='ones', choices=['ones', 'zeros'])
    args = parser.parse_args()

    device = settings.DEVICE
    logger.info(f"Device: {device}")
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = get_transforms(args.image_size)
    data_dir = Path(args.data_dir)

    # Load dataset
    if args.train_csv:
        train_ds = CheXpertDataset(args.train_csv, data_dir, train_tf, args.policy)
        if args.val_csv:
            val_ds = CheXpertDataset(args.val_csv, data_dir, val_tf, args.policy)
        else:
            t, v = int(len(train_ds)*0.8), len(train_ds) - int(len(train_ds)*0.8)
            train_ds, val_ds = random_split(train_ds, [t, v])
    elif (data_dir / 'train.csv').exists():
        logger.info("Found train.csv, using CheXpert format")
        train_ds = CheXpertDataset(data_dir / 'train.csv', data_dir, train_tf, args.policy)
        if (data_dir / 'valid.csv').exists():
            val_ds = CheXpertDataset(data_dir / 'valid.csv', data_dir, val_tf, args.policy)
        else:
            t = int(len(train_ds)*0.8)
            train_ds, val_ds = random_split(train_ds, [t, len(train_ds)-t])
    else:
        logger.info("No CSV found, using folder structure")
        full_ds = SimpleImageFolderDataset(data_dir, train_tf)
        if len(full_ds) == 0:
            logger.error("No data found! Provide CheXpert CSV or folder structure.")
            sys.exit(1)
        t = int(len(full_ds)*0.8)
        train_ds, val_ds = random_split(full_ds, [t, len(full_ds)-t])

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    if len(train_ds) == 0:
        logger.error("No training data!")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        model = ChestXrayDenseNet121.load_from_checkpoint(args.resume, num_classes=settings.NUM_CLASSES, device=str(device))
    else:
        model = ChestXrayDenseNet121(num_classes=settings.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_auc = 0.0
    history = []

    logger.info("=" * 60)
    logger.info(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | AUC: {val_auc:.4f} | {elapsed:.1f}s")

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_auc': val_auc, 'lr': optimizer.param_groups[0]['lr'], 'time': elapsed})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = val_auc
            model.save_checkpoint(str(output_dir / 'best_densenet121.pth'), optimizer=optimizer, epoch=epoch, metrics={'val_loss': val_loss, 'val_auc': val_auc})
            logger.info(f"  >> Saved best model (loss={val_loss:.4f}, AUC={val_auc:.4f})")

        model.save_checkpoint(str(output_dir / 'latest_densenet121.pth'), optimizer=optimizer, epoch=epoch, metrics={'val_loss': val_loss, 'val_auc': val_auc})

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Done! Best Val Loss: {best_val_loss:.4f}, AUC: {best_auc:.4f}")
    logger.info(f"Model: {output_dir / 'best_densenet121.pth'}")
    logger.info("Copy file best_densenet121.pth sang may khac de dung (khong can GPU).")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()