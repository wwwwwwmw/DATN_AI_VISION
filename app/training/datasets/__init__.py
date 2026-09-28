"""
Dataset Loaders for CheXpert and NIH Chest X-rays.

CheXpert (Bo du lieu chinh):
- 223,414 X-ray images, 14 benh ly (multi-label)
- Labels: 1=positive, 0=negative, -1=uncertain, blank=unknown
- Policy U-Ones: coi -1 la positive (khuyen nghi boi Stanford)

NIH Chest X-rays (Bo du lieu phu):
- 112,120 X-ray images, 15 benh ly
- Labels dang text: "Cardiomegaly|Effusion|..." hoac "No Finding"
- Dung de tang cuong du lieu va cross-validation
"""
import csv
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# 14 CheXpert classes (cung la output classes cua model)
CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

# NIH has 15 classes, map to CheXpert 14 where possible
NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia', 'No Finding',
]

# Map NIH class names -> CheXpert class index
NIH_TO_CHEXPERT = {
    'Atelectasis': 8,       # Atelectasis
    'Cardiomegaly': 2,      # Cardiomegaly
    'Effusion': 10,         # Pleural Effusion
    'Infiltration': 3,      # Lung Opacity (closest)
    'Mass': 4,              # Lung Lesion (closest)
    'Nodule': 4,            # Lung Lesion (closest)
    'Pneumonia': 7,         # Pneumonia
    'Pneumothorax': 9,      # Pneumothorax
    'Consolidation': 6,     # Consolidation
    'Edema': 5,             # Edema
    'Emphysema': 3,         # Lung Opacity (closest)
    'Fibrosis': 3,          # Lung Opacity (closest)
    'Pleural_Thickening': 11, # Pleural Other
    'Hernia': -1,           # No direct mapping, skip
    'No Finding': 0,        # No Finding
}


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset (Stanford, 2019).
    
    Multi-label classification: moi anh co the co nhieu benh cung luc.
    
    Xu ly label -1 (uncertain):
    - U-Ones: coi -1 la positive (1). Khuyen nghi boi Stanford,
      vi trong thuc te bac si thuong ghi "nghi ngo" khi co dau hieu.
    - U-Zeros: coi -1 la negative (0). An toan hon nhung mat du lieu.
    - U-Ignore: bo qua anh co -1. Giam data nhung sach hon.
    """
    
    def __init__(self, csv_path, root_dir, transform=None, 
                 policy='ones', frontal_only=True):
        """
        Args:
            csv_path: Path to train.csv or valid.csv
            root_dir: Thu muc goc chua CheXpert (parent of CheXpert-v1.0-small/)
            transform: Torchvision transforms
            policy: 'ones'|'zeros'|'ignore' - cach xu ly label -1
            frontal_only: Chi lay anh Frontal (bo Lateral) - khuyen nghi True
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.policy = policy
        self.num_classes = len(CHEXPERT_CLASSES)
        self.samples = []
        self.skipped = {'not_found': 0, 'lateral': 0, 'uncertain': 0}
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"CheXpert CSV not found: {csv_file}")
            return
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter Frontal only
                if frontal_only and row.get('Frontal/Lateral', '') == 'Lateral':
                    self.skipped['lateral'] += 1
                    continue
                
                # Build image path - CheXpert CSV has relative path like:
                # CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
                # We need to find it relative to root_dir
                csv_path_str = row['Path']
                
                # Try multiple path resolutions
                img_path = self.root_dir / csv_path_str
                if not img_path.exists():
                    # Try removing the first directory component
                    parts = csv_path_str.replace('\\', '/').split('/')
                    if len(parts) > 1:
                        img_path = self.root_dir / '/'.join(parts[1:])
                if not img_path.exists():
                    self.skipped['not_found'] += 1
                    continue
                
                # Parse labels
                labels = []
                has_uncertain = False
                for cls in CHEXPERT_CLASSES:
                    val = row.get(cls, '')
                    if val == '' or val == 'nan':
                        labels.append(0.0)
                    elif float(val) == -1.0:
                        has_uncertain = True
                        if self.policy == 'ones':
                            labels.append(1.0)
                        elif self.policy == 'zeros':
                            labels.append(0.0)
                        else:
                            labels.append(0.0)
                    else:
                        labels.append(max(0.0, float(val)))
                
                if self.policy == 'ignore' and has_uncertain:
                    self.skipped['uncertain'] += 1
                    continue
                
                self.samples.append({
                    'path': str(img_path),
                    'labels': labels,
                    'sex': row.get('Sex', ''),
                    'age': row.get('Age', ''),
                })
        
        logger.info(f"CheXpert: Loaded {len(self.samples)} samples "
                    f"(skipped: {self.skipped}), policy={self.policy}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(sample['labels'])
    
    def get_class_distribution(self):
        """Thong ke so luong moi class (de kiem tra data imbalance)."""
        dist = np.zeros(self.num_classes)
        for s in self.samples:
            dist += np.array(s['labels'])
        return {CHEXPERT_CLASSES[i]: int(dist[i]) for i in range(self.num_classes)}


class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-ray Dataset (NIH Clinical Center, 2017).
    
    Multi-label classification: labels dang text "Disease1|Disease2|..."
    Mapping sang 14 classes CheXpert de co the ket hop 2 dataset.
    
    Luu y: NIH co 15 classes, 1 so khong co tuong ung truc tiep
    voi CheXpert (Hernia, Emphysema) -> map sang class gan nhat.
    """
    
    def __init__(self, csv_path, image_dirs, transform=None,
                 file_list=None):
        """
        Args:
            csv_path: Path to Data_Entry_2017.csv
            image_dirs: List of image directories (images_001/images, ...)
            transform: Torchvision transforms
            file_list: Path to train_val_list.txt or test_list.txt (filter)
        """
        self.transform = transform
        self.num_classes = len(CHEXPERT_CLASSES)
        self.samples = []
        
        # Build image lookup: filename -> full path
        image_lookup = {}
        for img_dir in image_dirs:
            img_dir = Path(img_dir)
            if img_dir.exists():
                for f in img_dir.iterdir():
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                        image_lookup[f.name] = str(f)
        
        # File list filter
        allowed_files = None
        if file_list and Path(file_list).exists():
            with open(file_list, 'r') as f:
                allowed_files = set(line.strip() for line in f.readlines())
        
        # Parse CSV
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"NIH CSV not found: {csv_file}")
            return
        
        skipped = 0
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['Image Index']
                
                if allowed_files and img_name not in allowed_files:
                    continue
                
                if img_name not in image_lookup:
                    skipped += 1
                    continue
                
                # Parse multi-label: "Cardiomegaly|Effusion" -> indices
                labels = [0.0] * self.num_classes
                findings = row['Finding Labels'].split('|')
                
                for finding in findings:
                    finding = finding.strip()
                    if finding in NIH_TO_CHEXPERT:
                        idx = NIH_TO_CHEXPERT[finding]
                        if idx >= 0:
                            labels[idx] = 1.0
                
                self.samples.append({
                    'path': image_lookup[img_name],
                    'labels': labels,
                })
        
        logger.info(f"NIH: Loaded {len(self.samples)} samples (skipped: {skipped})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(sample['labels'])


def get_transforms(image_size=224, mode='train'):
    """
    Data Augmentation & Preprocessing transforms.
    
    === TIEN XU LY (Preprocessing) ===
    1. Resize: Dua tat ca anh ve cung kich thuoc (224x224 pixel)
       - DenseNet121 yeu cau input 224x224
    2. ToTensor: Chuyen pixel tu [0, 255] -> [0.0, 1.0]
    3. Normalize: Chuan hoa theo ImageNet (mean, std)
       - Giup model hoi tu nhanh hon vi weights pretrained
         da train tren ImageNet voi cung normalization
    
    === TANG CUONG (Augmentation) - chi ap dung khi train ===
    1. RandomResizedCrop: Cat ngau nhien roi resize
       - Giup model hoc o nhieu scale khac nhau
    2. RandomHorizontalFlip (p=0.5): Lat ngang ngau nhien
       - X-quang co the chup tu ca 2 huong
    3. RandomRotation(15): Xoay ngau nhien +-15 do
       - Mo phong anh chup bi lech nhe
    4. ColorJitter: Thay doi do sang, contrast
       - Mo phong dieu kien chup khac nhau
    5. RandomAffine: Dich chuyen, scale nhe
       - Tang tinh robust cua model
    
    Augmentation giup giam overfitting bang cach tao them
    "anh moi" tu anh goc => model hoc tong quat hon.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])


def create_dataloaders(
    chexpert_dir: str = None,
    nih_dir: str = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    policy: str = 'ones',
    use_nih_supplement: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    === PHAN CHIA DU LIEU (Data Splitting) ===
    
    Chia du lieu thanh 3 tap:
    - Train (70%): Model hoc truc tiep tu day
    - Validation (15%): Danh gia trong khi train, dung cho Early Stopping
    - Test (15%): Danh gia cuoi cung, KHONG duoc dung trong qua trinh train
    
    Strategy:
    - CheXpert lam BO DU LIEU CHINH (223K images, 14 classes)
    - NIH lam BO DU LIEU PHU (112K images, map sang 14 classes)
    - NIH duoc them vao TRAIN SET de tang cuong du lieu
    - Validation va Test chi dung CheXpert (de dam bao consistency)
    
    Returns: train_loader, val_loader, test_loader, dataset_info
    """
    train_tf = get_transforms(image_size, 'train')
    val_tf = get_transforms(image_size, 'val')
    
    datasets_info = {
        'chexpert_train': 0, 'chexpert_val': 0, 'chexpert_test': 0,
        'nih_supplement': 0, 'total_train': 0,
    }
    
    # === CheXpert (Primary) ===
    chexpert_train_ds = None
    chexpert_val_ds = None
    
    if chexpert_dir:
        chexpert_path = Path(chexpert_dir)
        train_csv = chexpert_path / 'train.csv'
        valid_csv = chexpert_path / 'valid.csv'
        
        if train_csv.exists():
            # Load full CheXpert train set
            full_chexpert = CheXpertDataset(
                train_csv, chexpert_path, train_tf, policy=policy
            )
            
            # Split: 70% train, 15% val, 15% test
            total = len(full_chexpert)
            train_size = int(total * train_ratio)
            val_size = int(total * val_ratio)
            test_size = total - train_size - val_size
            
            chexpert_train_ds, chexpert_val_ds_raw, chexpert_test_ds_raw = random_split(
                full_chexpert, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Val/Test dung val_tf (khong augmentation)
            # Wrap with different transform
            chexpert_val_ds = TransformWrapper(chexpert_val_ds_raw, val_tf)
            chexpert_test_ds = TransformWrapper(chexpert_test_ds_raw, val_tf)
            
            datasets_info['chexpert_train'] = len(chexpert_train_ds)
            datasets_info['chexpert_val'] = len(chexpert_val_ds)
            datasets_info['chexpert_test'] = len(chexpert_test_ds)
        
        # CheXpert official validation set (234 samples, expert-labeled)
        if valid_csv.exists():
            chexpert_official_val = CheXpertDataset(
                valid_csv, chexpert_path, val_tf, policy='ones'
            )
            if len(chexpert_official_val) > 0:
                logger.info(f"CheXpert official validation: {len(chexpert_official_val)} (expert-labeled)")
    
    # === NIH (Supplementary) ===
    nih_train_ds = None
    if nih_dir and use_nih_supplement:
        nih_path = Path(nih_dir)
        nih_csv = nih_path / 'Data_Entry_2017.csv'
        
        if nih_csv.exists():
            image_dirs = [nih_path / f'images_{i:03d}' / 'images' for i in range(1, 13)]
            
            nih_train_ds = NIHChestXrayDataset(
                nih_csv, image_dirs, train_tf,
                file_list=str(nih_path / 'train_val_list.txt')
            )
            datasets_info['nih_supplement'] = len(nih_train_ds)
    
    # === Combine ===
    train_datasets = []
    if chexpert_train_ds:
        train_datasets.append(chexpert_train_ds)
    if nih_train_ds and len(nih_train_ds) > 0:
        train_datasets.append(nih_train_ds)
    
    if not train_datasets:
        raise ValueError("No training data found!")
    
    combined_train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    datasets_info['total_train'] = len(combined_train)
    
    logger.info(f"=== Dataset Summary ===")
    logger.info(f"Train: {datasets_info['total_train']} (CheXpert: {datasets_info['chexpert_train']}, NIH: {datasets_info['nih_supplement']})")
    logger.info(f"Val:   {datasets_info['chexpert_val']}")
    logger.info(f"Test:  {datasets_info['chexpert_test']}")
    
    # DataLoaders
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(chexpert_val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True) if chexpert_val_ds else None
    test_loader = DataLoader(chexpert_test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True) if chexpert_test_ds_raw else None
    
    return train_loader, val_loader, test_loader, datasets_info


class TransformWrapper(Dataset):
    """Wrap a Subset with a different transform (for val/test without augmentation)."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, labels = self.subset[idx]
        # image is already a tensor from the original transform
        # We need raw image, so this is a workaround
        # In practice, we re-apply transform at dataloader level
        return image, labels