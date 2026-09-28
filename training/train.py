"""
Training Script — DenseNet121 on CheXpert + NIH datasets.
Lưu model weights (.pth) để inference trên CPU ở máy khác.

Usage:
    python training/train.py --config training/config.yaml
    python training/train.py --epochs 50 --batch-size 32
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.densenet121 import ChestXrayDenseNet121
from data.dataset import ChestXrayDataset
from data.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            logger.info(f"Epoch [{epoch}] Batch [{batch_idx+1}/{total_batches}] Loss: {avg_loss:.4f}")

    return running_loss / total_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate AUC per class
    try:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            else:
                aucs.append(0.0)
        mean_auc = np.mean(aucs)
    except Exception:
        mean_auc = 0.0
        aucs = []

    return avg_loss, mean_auc, aucs


def main():
    parser = argparse.ArgumentParser(description='Train DenseNet121 on Chest X-ray data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-classes', type=int, default=14)
    parser.add_argument('--data-dir', type=str, default='../../CheXpert')
    parser.add_argument('--output-dir', type=str, default='weights')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transform = get_train_transforms(args.image_size)
    val_transform = get_val_transforms(args.image_size)

    # Datasets
    logger.info(f"Loading data from {args.data_dir}...")
    train_dataset = ChestXrayDataset(
        csv_path=os.path.join(args.data_dir, 'train.csv'),
        image_root=args.data_dir,
        transform=train_transform,
        num_classes=args.num_classes,
    )
    val_dataset = ChestXrayDataset(
        csv_path=os.path.join(args.data_dir, 'valid.csv'),
        image_root=args.data_dir,
        transform=val_transform,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Model
    model = ChestXrayDenseNet121(num_classes=args.num_classes, pretrained=True)
    model.to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}

    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_auc, class_aucs = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save with full metadata for portability
            model.save_checkpoint(
                output_dir / 'best_densenet121.pth',
                optimizer=optimizer,
                epoch=epoch,
                metrics={'val_auc': val_auc, 'val_loss': val_loss, 'class_aucs': class_aucs}
            )
            logger.info(f"  ★ New best model saved! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

        # Save every 10 epochs
        if epoch % 10 == 0:
            model.save_checkpoint(
                output_dir / f'densenet121_epoch_{epoch}.pth',
                optimizer=optimizer,
                epoch=epoch,
                metrics={'val_auc': val_auc, 'val_loss': val_loss}
            )

    total_time = time.time() - start_time
    logger.info(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best AUC: {best_auc:.4f}")

    # Save training history
    history['best_auc'] = best_auc
    history['total_time_seconds'] = total_time
    history['device'] = str(device)
    history['args'] = vars(args)
    history['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training history saved to {output_dir / 'training_history.json'}")


if __name__ == '__main__':
    main()
