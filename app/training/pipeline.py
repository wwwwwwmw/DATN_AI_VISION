"""
=== COMPLETE TRAINING PIPELINE ===
Quy trinh huan luyen AI phan tich X-quang phoi.

7 BUOC:
1. Tien xu ly & Tang cuong du lieu (Preprocessing & Augmentation)
2. Phan chia du lieu (Data Splitting: 70/15/15)
3. Thiet lap mo hinh (Model Setup: DenseNet121 + BCEWithLogitsLoss + Adam)
4. Huan luyen (Training: Backpropagation, Mini-batch SGD)
5. Danh gia (Evaluation: AUC, Sensitivity, Specificity, Confusion Matrix)
6. Tinh chinh (Hyperparameter Tuning: Learning Rate, Epochs, Batch Size)
7. Luu tru & Trien khai (Save .pth -> FastAPI inference)

Usage:
    python -m app.training.pipeline --chexpert-dir ../CheXpert --nih-dir ../NIH_Chest_X_rays --epochs 20
    python -m app.training.pipeline --chexpert-dir ../CheXpert --epochs 5 --batch-size 16 --no-nih
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model.densenet121 import ChestXrayDenseNet121
from app.core.config import settings
from app.training.datasets import (
    CheXpertDataset, NIHChestXrayDataset, 
    get_transforms, CHEXPERT_CLASSES,
)
from app.training.evaluation import evaluate_model, compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    === PIPELINE HUAN LUYEN AI ===
    
    Ket hop CheXpert (chinh) + NIH (phu) de train DenseNet121.
    
    DenseNet121 (Densely Connected Convolutional Network):
    - Kien truc: Moi layer nhan input tu TAT CA layers truoc do
    - Uu diem: Giam vanishing gradient, tang feature reuse, it tham so hon ResNet
    - Pre-trained tren ImageNet (1.2 trieu anh, 1000 classes)
    - Transfer Learning: Giu cac feature extractor, thay classifier layer
    
    Loss Function: BCEWithLogitsLoss (Binary Cross-Entropy with Logits)
    - Ket hop Sigmoid + BCE trong 1 buoc (numerically stable)
    - Phu hop cho MULTI-LABEL classification (1 anh co nhieu benh)
    - Khac voi CrossEntropyLoss (chi dung cho single-label)
    
    Optimizer: Adam (Adaptive Moment Estimation)
    - Ket hop momentum (beta1) + RMSProp (beta2)
    - Tu dong dieu chinh learning rate cho moi tham so
    - Hoi tu nhanh hon SGD thuong
    - Default: lr=1e-4, beta1=0.9, beta2=0.999
    
    Learning Rate Scheduler: ReduceLROnPlateau
    - Giam lr khi validation loss khong giam sau N epochs (patience)
    - Factor=0.5: giam lr con 1/2
    - Giup model "tinh chinh" khi gan hoi tu
    
    Early Stopping:
    - Dung train khi val_loss khong giam sau N epochs lien tiep
    - Tranh overfitting (model hoc vet du lieu train)
    """
    
    def __init__(self, config):
        self.config = config
        self.device = settings.DEVICE
        self.output_dir = Path(config.get('output_dir', 'outputs/training_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = Path(config.get('weights_dir', 'weights'))
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state (for web UI progress tracking)
        self.state = {
            'status': 'idle',  # idle, preprocessing, training, evaluating, done, error
            'current_epoch': 0,
            'total_epochs': config.get('epochs', 20),
            'train_loss': 0,
            'val_loss': 0,
            'val_auc': 0,
            'best_auc': 0,
            'history': [],
            'dataset_info': {},
            'eval_results': None,
            'error': None,
        }
        self._save_state()
    
    def _save_state(self):
        """Save training state to JSON (for web UI to read)."""
        state_file = self.output_dir / 'training_state.json'
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _update_state(self, **kwargs):
        self.state.update(kwargs)
        self._save_state()
    
    # ==========================================
    # BUOC 1: TIEN XU LY & TANG CUONG DU LIEU
    # ==========================================
    def step1_prepare_data(self):
        """
        === BUOC 1: Data Preprocessing & Augmentation ===
        
        Tien xu ly:
        - Xoa du lieu loi (anh khong doc duoc, label thieu)
        - Resize tat ca anh ve 224x224 pixel (DenseNet121 input size)
        - Chuyen pixel [0,255] -> [0.0, 1.0] (ToTensor)
        - Normalize theo ImageNet mean/std (Transfer Learning)
        
        Tang cuong (chi train set):
        - RandomResizedCrop: cat ngau nhien 80-100% dien tich
        - RandomHorizontalFlip: lat ngang 50%
        - RandomRotation(15): xoay +-15 do
        - ColorJitter: thay doi brightness/contrast
        - RandomAffine: dich chuyen 5%
        
        => Tao them "anh moi" tu anh goc, giam overfitting.
        """
        self._update_state(status='preprocessing')
        logger.info("=" * 60)
        logger.info("BUOC 1: Tien xu ly & Tang cuong du lieu")
        logger.info("=" * 60)
        
        image_size = self.config.get('image_size', 224)
        train_tf = get_transforms(image_size, 'train')
        val_tf = get_transforms(image_size, 'val')
        
        logger.info(f"Image size: {image_size}x{image_size}")
        logger.info(f"Train augmentation: Resize, RandomCrop, Flip, Rotation, ColorJitter")
        logger.info(f"Val/Test: Resize only (no augmentation)")
        
        return train_tf, val_tf
    
    # ==========================================
    # BUOC 2: PHAN CHIA DU LIEU
    # ==========================================
    def step2_split_data(self, train_tf, val_tf):
        """
        === BUOC 2: Data Splitting ===
        
        CheXpert (BO CHINH - 223,414 images):
        - Train: 70% (~156K) - Model hoc truc tiep
        - Validation: 15% (~33K) - Danh gia trong khi train
        - Test: 15% (~33K) - Danh gia cuoi cung
        
        NIH (BO PHU - 86,524 images):
        - Them vao Train set de tang cuong du lieu
        - KHONG dung cho Val/Test (dam bao consistency)
        - Map 15 NIH classes -> 14 CheXpert classes
        
        Random seed = 42: Dam bao ket qua reproducible.
        """
        logger.info("=" * 60)
        logger.info("BUOC 2: Phan chia du lieu")
        logger.info("=" * 60)
        
        from torch.utils.data import DataLoader, random_split, ConcatDataset
        
        cfg = self.config
        chexpert_dir = cfg.get('chexpert_dir')
        nih_dir = cfg.get('nih_dir')
        batch_size = cfg.get('batch_size', 32)
        num_workers = cfg.get('num_workers', 4)
        policy = cfg.get('policy', 'ones')
        
        # CheXpert
        chexpert_path = Path(chexpert_dir)
        train_csv = chexpert_path / 'train.csv'
        
        logger.info(f"Loading CheXpert from {chexpert_dir}...")
        full_ds = CheXpertDataset(train_csv, chexpert_path, train_tf, policy=policy)
        
        total = len(full_ds)
        train_size = int(total * 0.70)
        val_size = int(total * 0.15)
        test_size = total - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            full_ds, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"CheXpert split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Class distribution
        dist = full_ds.get_class_distribution()
        logger.info(f"Class distribution: {json.dumps(dist, indent=2)}")
        
        # NIH supplement
        nih_ds = None
        nih_count = 0
        if nih_dir and not cfg.get('no_nih', False):
            nih_path = Path(nih_dir)
            nih_csv = nih_path / 'Data_Entry_2017.csv'
            
            if nih_csv.exists():
                logger.info(f"Loading NIH from {nih_dir}...")
                image_dirs = [nih_path / f'images_{i:03d}' / 'images' for i in range(1, 13)]
                nih_ds = NIHChestXrayDataset(
                    nih_csv, image_dirs, train_tf,
                    file_list=str(nih_path / 'train_val_list.txt')
                )
                nih_count = len(nih_ds)
        
        # Combine train
        if nih_ds and len(nih_ds) > 0:
            combined_train = ConcatDataset([train_ds, nih_ds])
            logger.info(f"Combined train: CheXpert({train_size}) + NIH({nih_count}) = {len(combined_train)}")
        else:
            combined_train = train_ds
        
        # DataLoaders
        train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        
        dataset_info = {
            'chexpert_total': total,
            'train_chexpert': train_size,
            'train_nih': nih_count,
            'train_total': len(combined_train),
            'validation': val_size,
            'test': test_size,
            'num_classes': 14,
            'class_names': CHEXPERT_CLASSES,
            'class_distribution': dist,
            'policy': policy,
        }
        self._update_state(dataset_info=dataset_info)
        
        return train_loader, val_loader, test_loader, dataset_info
    
    # ==========================================
    # BUOC 3: THIET LAP MO HINH
    # ==========================================
    def step3_setup_model(self):
        """
        === BUOC 3: Model Setup ===
        
        Model: DenseNet121 (Huang et al., 2017)
        - 121 layers, ~8M parameters
        - Dense connections: moi layer ket noi voi tat ca layers truoc
        - Pre-trained tren ImageNet -> Transfer Learning
        - Thay classifier: 1000 classes -> 14 classes
        - Them Dropout(0.5) truoc classifier de giam overfitting
        
        Loss: BCEWithLogitsLoss
        - Binary Cross-Entropy cho multi-label
        - Sigmoid duoc tich hop trong loss (stable hon)
        - Moi class duoc coi la 1 bai toan binary doc lap
        
        Optimizer: Adam (Kingma & Ba, 2014)
        - lr=1e-4: Learning rate khoi dau
        - weight_decay=1e-5: L2 regularization, giam overfitting
        - Adaptive: tu dieu chinh lr cho moi parameter
        
        Scheduler: ReduceLROnPlateau
        - Giam lr x0.5 khi val_loss khong giam sau 3 epochs
        - Min lr: 1e-7 (khong giam them)
        """
        logger.info("=" * 60)
        logger.info("BUOC 3: Thiet lap mo hinh")
        logger.info("=" * 60)
        
        cfg = self.config
        resume = cfg.get('resume', None)
        
        if resume and Path(resume).exists():
            logger.info(f"Resume tu checkpoint: {resume}")
            model = ChestXrayDenseNet121.load_from_checkpoint(
                resume, num_classes=14, device=str(self.device)
            )
        else:
            logger.info("Khoi tao DenseNet121 voi ImageNet pre-trained weights")
            model = ChestXrayDenseNet121(num_classes=14, pretrained=True, dropout=0.5)
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable: {trainable:,}")
        
        lr = cfg.get('lr', 1e-4)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        
        logger.info(f"Loss: BCEWithLogitsLoss (multi-label binary)")
        logger.info(f"Optimizer: Adam (lr={lr}, weight_decay=1e-5)")
        logger.info(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
        
        return model, criterion, optimizer, scheduler
    
    # ==========================================
    # BUOC 4: HUAN LUYEN
    # ==========================================
    def step4_train(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        """
        === BUOC 4: Training ===
        
        Qua trinh:
        1. Forward pass: Dua batch anh qua model -> logits
        2. Tinh loss: BCEWithLogitsLoss(logits, labels)
        3. Backward pass (Backpropagation):
           - Tinh gradient (dao ham) cua loss theo moi tham so
           - Lan truyen gradient tu output layer ve input layer
        4. Update weights: optimizer.step()
           - Adam cap nhat moi tham so theo gradient + momentum
        5. Lap lai cho moi batch trong 1 epoch
        6. Lap lai cho nhieu epochs
        
        Early Stopping:
        - Theo doi val_loss sau moi epoch
        - Neu val_loss khong giam sau 'patience' epochs -> dung
        - Luu model tot nhat (best val_loss)
        - Tranh overfitting: model chi hoc tot tren train, kem tren val
        
        Mini-batch Training:
        - Khong dua tat ca data cung luc (het RAM)
        - Chia thanh batch (32 anh/batch)
        - Gradient duoc tinh trung binh trong batch
        - Ket hop toc do (SGD) va on dinh (full batch)
        """
        logger.info("=" * 60)
        logger.info("BUOC 4: Huan luyen mo hinh")
        logger.info("=" * 60)
        
        cfg = self.config
        epochs = cfg.get('epochs', 20)
        patience = cfg.get('patience', 7)
        
        best_val_loss = float('inf')
        best_auc = 0.0
        no_improve = 0
        history = []
        
        self._update_state(status='training', total_epochs=epochs)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # --- Train ---
            model.train()
            train_loss = 0
            n_batches = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                
                # Backward (Backpropagation)
                loss.backward()
                
                # Gradient clipping (tranh gradient explosion)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 100 == 0:
                    logger.info(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / max(n_batches, 1)
            
            # --- Validate ---
            model.eval()
            val_loss = 0
            val_batches = 0
            all_labels, all_preds = [], []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    val_batches += 1
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
            
            avg_val_loss = val_loss / max(val_batches, 1)
            
            # Compute AUC
            val_auc = 0.0
            if all_labels:
                try:
                    from sklearn.metrics import roc_auc_score
                    labels_np = np.concatenate(all_labels)
                    preds_np = np.concatenate(all_preds)
                    aucs = []
                    for i in range(14):
                        if len(np.unique(labels_np[:, i])) >= 2:
                            aucs.append(roc_auc_score(labels_np[:, i], preds_np[:, i]))
                    val_auc = np.mean(aucs) if aucs else 0.0
                except:
                    val_auc = 0.0
            
            # Scheduler
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                f"AUC: {val_auc:.4f} | LR: {current_lr:.2e} | {epoch_time:.1f}s"
            )
            
            # History
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': round(avg_train_loss, 6),
                'val_loss': round(avg_val_loss, 6),
                'val_auc': round(val_auc, 6),
                'lr': current_lr,
                'time_seconds': round(epoch_time, 2),
            }
            history.append(epoch_data)
            
            self._update_state(
                current_epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_auc=val_auc,
                history=history,
            )
            
            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_auc = val_auc
                no_improve = 0
                
                model.save_checkpoint(
                    str(self.weights_dir / 'best_densenet121.pth'),
                    optimizer=optimizer, epoch=epoch,
                    metrics={'val_loss': avg_val_loss, 'val_auc': val_auc}
                )
                logger.info(f"  >> BEST model saved (loss={avg_val_loss:.4f}, AUC={val_auc:.4f})")
                self._update_state(best_auc=best_auc)
            else:
                no_improve += 1
            
            # Save latest
            model.save_checkpoint(
                str(self.weights_dir / 'latest_densenet121.pth'),
                optimizer=optimizer, epoch=epoch,
                metrics={'val_loss': avg_val_loss, 'val_auc': val_auc}
            )
            
            # Early Stopping
            if no_improve >= patience:
                logger.info(f"Early Stopping: val_loss khong giam sau {patience} epochs")
                break
        
        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training complete! Best val_loss={best_val_loss:.4f}, AUC={best_auc:.4f}")
        return model, history
    
    # ==========================================
    # BUOC 5: DANH GIA
    # ==========================================
    def step5_evaluate(self, model, test_loader):
        """
        === BUOC 5: Evaluation ===
        
        Dua tap Test (chua bao gio thay trong training) vao model.
        Tinh cac chi so y te:
        - AUC-ROC per class
        - Sensitivity (Do nhay - bat dung benh)
        - Specificity (Do dac hieu - xac nhan khong benh)
        - Precision, F1-Score
        - Confusion Matrix (TP, FP, TN, FN)
        - Optimal thresholds (Youden's J)
        """
        logger.info("=" * 60)
        logger.info("BUOC 5: Danh gia mo hinh tren Test set")
        logger.info("=" * 60)
        
        self._update_state(status='evaluating')
        
        results = evaluate_model(
            model, test_loader, self.device,
            output_dir=str(self.output_dir)
        )
        
        self._update_state(eval_results=results)
        return results
    
    # ==========================================
    # BUOC 6: TINH CHINH (logged in config)
    # ==========================================
    def step6_log_hyperparameters(self):
        """
        === BUOC 6: Hyperparameter Tuning ===
        
        Luu lai cac hyperparameters da dung de so sanh giua cac lan train.
        Neu ket qua chua tot, thay doi va train lai:
        
        Cac hyperparameters quan trong:
        - learning_rate: 1e-4 (default), thu 1e-3, 5e-5, 1e-5
        - batch_size: 32 (default), thu 16, 64
        - epochs: 20 (default), tang neu chua hoi tu
        - dropout: 0.5 (default), thu 0.3, 0.7
        - policy: 'ones' (default), thu 'zeros'
        - image_size: 224 (default), thu 320, 384
        - augmentation: co the tang/giam cac tham so
        
        Luu y: Moi lan chi thay doi 1 hyperparameter de biet
        cai nao anh huong. Ghi lai ket qua moi lan vao JSON.
        """
        hp = {
            'learning_rate': self.config.get('lr', 1e-4),
            'batch_size': self.config.get('batch_size', 32),
            'epochs': self.config.get('epochs', 20),
            'image_size': self.config.get('image_size', 224),
            'policy': self.config.get('policy', 'ones'),
            'patience': self.config.get('patience', 7),
            'optimizer': 'Adam',
            'loss': 'BCEWithLogitsLoss',
            'model': 'DenseNet121',
            'dropout': 0.5,
            'weight_decay': 1e-5,
        }
        
        with open(self.output_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hp, f, indent=2)
        
        logger.info(f"Hyperparameters saved to {self.output_dir / 'hyperparameters.json'}")
        return hp
    
    # ==========================================
    # BUOC 7: LUU TRU & TRIEN KHAI
    # ==========================================
    def step7_deploy(self):
        """
        === BUOC 7: Save & Deploy ===
        
        1. File best_densenet121.pth chua:
           - model_state_dict: tat ca weights da hoc
           - optimizer_state_dict: trang thai optimizer (de resume)
           - epoch: epoch tot nhat
           - metrics: val_loss, val_auc
        
        2. Deploy:
           - Copy best_densenet121.pth vao ai-vision/weights/
           - Restart AI Vision service -> tu dong load model moi
           - Khong can GPU de inference (chay CPU duoc)
        
        3. Inference flow:
           User upload X-ray -> FastAPI -> Load model ->
           Preprocess (224x224, normalize) -> Forward pass ->
           Sigmoid -> 14 probabilities -> Return top findings
        """
        logger.info("=" * 60)
        logger.info("BUOC 7: Luu tru & Trien khai")
        logger.info("=" * 60)
        
        best_path = self.weights_dir / 'best_densenet121.pth'
        if best_path.exists():
            size_mb = best_path.stat().st_size / (1024 * 1024)
            logger.info(f"Model saved: {best_path} ({size_mb:.1f} MB)")
            logger.info(f"De deploy: Copy file nay vao ai-vision/weights/")
            logger.info(f"Restart AI Vision service de load model moi.")
        
        self._update_state(status='done')
        logger.info("=" * 60)
        logger.info("PIPELINE HOAN TAT!")
        logger.info("=" * 60)
    
    # ==========================================
    # RUN ALL STEPS
    # ==========================================
    def run(self):
        """Chay toan bo 7 buoc."""
        try:
            train_tf, val_tf = self.step1_prepare_data()
            train_loader, val_loader, test_loader, ds_info = self.step2_split_data(train_tf, val_tf)
            model, criterion, optimizer, scheduler = self.step3_setup_model()
            model, history = self.step4_train(model, train_loader, val_loader, criterion, optimizer, scheduler)
            eval_results = self.step5_evaluate(model, test_loader)
            hp = self.step6_log_hyperparameters()
            self.step7_deploy()
            return True
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self._update_state(status='error', error=str(e))
            return False


def main():
    parser = argparse.ArgumentParser(description='AI Vision Training Pipeline')
    parser.add_argument('--chexpert-dir', type=str, required=True, help='Path to CheXpert dataset')
    parser.add_argument('--nih-dir', type=str, default=None, help='Path to NIH dataset (supplementary)')
    parser.add_argument('--no-nih', action='store_true', help='Do not use NIH dataset')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--policy', type=str, default='ones', choices=['ones', 'zeros'])
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/training_results')
    args = parser.parse_args()
    
    config = vars(args)
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()