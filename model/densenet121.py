"""
DenseNet121 — Custom model cho phân loại bệnh lý phổi.
"""
import torch
import torch.nn as nn
from torchvision import models


class ChestXrayDenseNet121(nn.Module):
    """
    DenseNet121 pretrained trên ImageNet,
    thay classifier layer cho multi-label classification.
    """

    def __init__(self, num_classes=14, pretrained=True, dropout=0.5):
        super().__init__()

        # Load DenseNet121 pretrained
        if pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.densenet = models.densenet121(weights=weights)
        else:
            self.densenet = models.densenet121(weights=None)

        # Get number of features from original classifier
        num_features = self.densenet.classifier.in_features

        # Replace classifier
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.
        Returns logits (for training with BCEWithLogitsLoss)
        """
        logits = self.densenet(x)
        return logits

    def predict(self, x):
        """
        Predict probabilities (for inference).
        Returns probabilities after sigmoid.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = self.sigmoid(logits)
        return probs

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, num_classes=14, device='cpu'):
        """
        Load model từ checkpoint .pth file.
        Hỗ trợ chạy trên CPU (không cần GPU).
        """
        model = cls(num_classes=num_classes, pretrained=False)

        # Load checkpoint (map_location cho CPU inference)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(device),
            weights_only=False,
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model

    def save_checkpoint(self, path, optimizer=None, epoch=None, metrics=None):
        """Lưu checkpoint (model weights + training state)."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_classes': self.densenet.classifier[1].out_features,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
