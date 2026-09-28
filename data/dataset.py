"""
Chest X-ray Dataset — CheXpert format.
Handles uncertain labels (U-Ones, U-Zeros, U-Ignore).
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset cho CheXpert format CSV.
    
    Handles:
    - Multi-label classification (14 classes)
    - Uncertain labels (-1.0) → configurable strategy
    - Frontal-only filtering
    - Missing values
    """

    # CheXpert label columns (14 classes)
    LABEL_COLUMNS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices',
    ]

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        num_classes: int = 14,
        uncertain_strategy: str = 'u_ones',
        frontal_only: bool = True,
    ):
        """
        Args:
            csv_path: Path to train.csv or valid.csv
            image_root: Root directory for image paths
            transform: torchvision transforms
            num_classes: Number of classes to use
            uncertain_strategy: 'u_ones' | 'u_zeros' | 'u_ignore'
            frontal_only: Only use frontal view images
        """
        self.image_root = image_root
        self.transform = transform
        self.num_classes = num_classes
        self.uncertain_strategy = uncertain_strategy

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Filter frontal only
        if frontal_only and 'Frontal/Lateral' in self.df.columns:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # Use only available label columns
        available_columns = [c for c in self.LABEL_COLUMNS[:num_classes] if c in self.df.columns]
        self.label_columns = available_columns

        # Fill NaN with 0
        self.df[self.label_columns] = self.df[self.label_columns].fillna(0.0)

        # Handle uncertain labels (-1.0)
        if uncertain_strategy == 'u_ones':
            # -1 → 1 (positive) — best for Atelectasis, Edema, Pleural Effusion
            self.df[self.label_columns] = self.df[self.label_columns].replace(-1.0, 1.0)
        elif uncertain_strategy == 'u_zeros':
            # -1 → 0 (negative)
            self.df[self.label_columns] = self.df[self.label_columns].replace(-1.0, 0.0)
        elif uncertain_strategy == 'u_ignore':
            # Remove rows with any -1 label
            mask = ~(self.df[self.label_columns] == -1.0).any(axis=1)
            self.df = self.df[mask]

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image path
        img_path = row.iloc[0]  # First column = Path
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return black image if file is corrupted
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Labels
        labels = row[self.label_columns].values.astype(np.float32)
        labels = np.clip(labels, 0.0, 1.0)

        return image, labels
