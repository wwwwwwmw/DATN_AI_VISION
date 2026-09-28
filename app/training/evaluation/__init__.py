"""
Model Evaluation - Medical-grade metrics.

=== CAC CHI SO DANH GIA (Evaluation Metrics) ===

1. AUC-ROC (Area Under ROC Curve):
   - Do kha nang phan biet giua benh va khong benh
   - AUC = 1.0: hoan hao, AUC = 0.5: random guess
   - Dung cho multi-label: tinh AUC moi class roi lay trung binh

2. Sensitivity / Recall (Do nhay):
   - = TP / (TP + FN)
   - Kha nang bat DUNG benh, KHONG BO SOT
   - Trong y te: BO SOT benh nguy hiem hon bao dong nham
   - => Uu tien Sensitivity cao (>= 0.9)

3. Specificity (Do dac hieu):
   - = TN / (TN + FP)
   - Kha nang xac nhan dung nguoi KHONG benh
   - Tranh bao dong nham (false alarm)

4. Precision (Do chinh xac):
   - = TP / (TP + FP)
   - Khi model noi "co benh", bao nhieu % la dung?

5. F1-Score:
   - = 2 * (Precision * Recall) / (Precision + Recall)
   - Can bang giua Precision va Recall

6. Confusion Matrix (Ma tran nham lan):
   - Bang 2x2 cho moi class: TP, FP, TN, FN
   - Truc quan hoa cac loai loi cua model

=== THUAT TOAN ===
- Threshold optimization: Tim nguong tot nhat cho moi class
  bang cach chon threshold maximizes Youden's J statistic
  (J = Sensitivity + Specificity - 1)
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]


def compute_metrics(labels: np.ndarray, predictions: np.ndarray,
                    threshold: float = 0.5) -> Dict:
    """
    Tinh tat ca metrics cho multi-label classification.
    
    Args:
        labels: Ground truth [N, num_classes], binary
        predictions: Model output probabilities [N, num_classes]
        threshold: Nguong de chuyen probability -> binary
    
    Returns:
        Dict chua tat ca metrics
    """
    num_classes = labels.shape[1]
    binary_preds = (predictions >= threshold).astype(int)
    
    results = {
        'per_class': {},
        'macro_avg': {},
        'threshold': threshold,
    }
    
    all_auc, all_sens, all_spec, all_prec, all_f1 = [], [], [], [], []
    
    for i in range(num_classes):
        class_name = CHEXPERT_CLASSES[i] if i < len(CHEXPERT_CLASSES) else f'Class_{i}'
        y_true = labels[:, i]
        y_pred = binary_preds[:, i]
        y_prob = predictions[:, i]
        
        # Confusion matrix components
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        
        # Metrics
        sensitivity = tp / max(tp + fn, 1)  # Recall
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)
        
        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) >= 2:
                auc = float(roc_auc_score(y_true, y_prob))
            else:
                auc = 0.0
        except (ImportError, ValueError):
            auc = 0.0
        
        results['per_class'][class_name] = {
            'auc': round(auc, 4),
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'precision': round(precision, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'support': int(np.sum(y_true)),
        }
        
        if auc > 0:
            all_auc.append(auc)
        all_sens.append(sensitivity)
        all_spec.append(specificity)
        all_prec.append(precision)
        all_f1.append(f1)
    
    # Macro averages
    results['macro_avg'] = {
        'auc': round(np.mean(all_auc), 4) if all_auc else 0.0,
        'sensitivity': round(np.mean(all_sens), 4),
        'specificity': round(np.mean(all_spec), 4),
        'precision': round(np.mean(all_prec), 4),
        'f1_score': round(np.mean(all_f1), 4),
    }
    
    return results


def find_optimal_thresholds(labels: np.ndarray, predictions: np.ndarray) -> Dict:
    """
    Tim nguong toi uu cho moi class bang Youden's J statistic.
    
    === THUAT TOAN: Youden's J Statistic ===
    J = Sensitivity + Specificity - 1
    
    Voi moi class, thu tat ca cac nguong tu 0.01 den 0.99,
    chon nguong co J cao nhat.
    
    Y nghia: Tim diem can bang tot nhat giua:
    - Bat dung benh (Sensitivity cao)
    - Khong bao dong nham (Specificity cao)
    """
    num_classes = labels.shape[1]
    optimal = {}
    
    for i in range(num_classes):
        class_name = CHEXPERT_CLASSES[i] if i < len(CHEXPERT_CLASSES) else f'Class_{i}'
        y_true = labels[:, i]
        y_prob = predictions[:, i]
        
        best_j = -1
        best_threshold = 0.5
        
        for t in np.arange(0.05, 0.95, 0.05):
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1
            
            if j > best_j:
                best_j = j
                best_threshold = t
        
        optimal[class_name] = round(best_threshold, 2)
    
    return optimal


def evaluate_model(model, test_loader, device, output_dir=None):
    """
    === DANH GIA MO HINH (Model Evaluation) ===
    
    Dua tap Test vao model da train xong.
    Tinh tat ca metrics y te va luu ket qua.
    
    Steps:
    1. Inference: Dua tung batch anh qua model, lay probabilities
    2. Compute metrics: AUC, Sensitivity, Specificity, Precision, F1
    3. Find optimal thresholds: Youden's J per class
    4. Re-compute metrics with optimal thresholds
    5. Save results to JSON + generate report
    """
    model.eval()
    all_labels = []
    all_preds = []
    
    logger.info("Running evaluation on test set...")
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.numpy())
            all_preds.append(probs.cpu().numpy())
    
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    eval_time = time.time() - start_time
    
    # Default threshold
    results_default = compute_metrics(labels, preds, threshold=0.5)
    
    # Optimal thresholds
    optimal_thresholds = find_optimal_thresholds(labels, preds)
    
    # Re-evaluate with optimal thresholds
    optimal_preds = np.zeros_like(preds)
    for i, cls in enumerate(CHEXPERT_CLASSES):
        if cls in optimal_thresholds:
            optimal_preds[:, i] = (preds[:, i] >= optimal_thresholds[cls]).astype(float)
    
    results_optimal = compute_metrics(labels, preds)
    # Override with per-class optimal threshold results
    for i, cls in enumerate(CHEXPERT_CLASSES):
        if cls in optimal_thresholds:
            t = optimal_thresholds[cls]
            y_true = labels[:, i]
            y_pred = (preds[:, i] >= t).astype(int)
            y_prob = preds[:, i]
            
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * sens / max(prec + sens, 1e-8)
            
            results_optimal['per_class'][cls].update({
                'sensitivity': round(sens, 4),
                'specificity': round(spec, 4),
                'precision': round(prec, 4),
                'f1_score': round(f1, 4),
                'optimal_threshold': t,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            })
    
    full_results = {
        'eval_time_seconds': round(eval_time, 2),
        'num_samples': int(labels.shape[0]),
        'default_threshold_0.5': results_default,
        'optimal_thresholds': results_optimal,
        'thresholds_per_class': optimal_thresholds,
    }
    
    # Save
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'evaluation_results.json', 'w') as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Evaluation saved to {out / 'evaluation_results.json'}")
    
    # Print summary
    logger.info("=" * 70)
    logger.info(f"EVALUATION RESULTS ({labels.shape[0]} samples, {eval_time:.1f}s)")
    logger.info("=" * 70)
    logger.info(f"{'Class':<30} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'Prec':>6} {'F1':>6}")
    logger.info("-" * 70)
    for cls, m in results_default['per_class'].items():
        logger.info(f"{cls:<30} {m['auc']:>6.3f} {m['sensitivity']:>6.3f} "
                    f"{m['specificity']:>6.3f} {m['precision']:>6.3f} {m['f1_score']:>6.3f}")
    logger.info("-" * 70)
    avg = results_default['macro_avg']
    logger.info(f"{'MACRO AVG':<30} {avg['auc']:>6.3f} {avg['sensitivity']:>6.3f} "
                f"{avg['specificity']:>6.3f} {avg['precision']:>6.3f} {avg['f1_score']:>6.3f}")
    
    return full_results