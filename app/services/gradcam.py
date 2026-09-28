"""
Grad-CAM — Gradient-weighted Class Activation Mapping.
Tạo heatmap hiển thị vùng phổi mà model chú ý.
"""
import torch
import numpy as np
import cv2
from pathlib import Path


class GradCAM:
    """
    Grad-CAM cho DenseNet121.
    Tạo heatmap hiển thị vùng ảnh đóng góp nhiều nhất vào dự đoán.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: ChestXrayDenseNet121 instance
            target_layer: Layer để lấy gradients (default: last conv layer)
        """
        self.model = model
        self.model.eval()

        # Default: denseblock4 (last dense block)
        if target_layer is None:
            self.target_layer = model.densenet.features.denseblock4
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor [1, 3, 224, 224]
            class_idx: Target class index (None = highest predicted class)

        Returns:
            heatmap: numpy array [H, W] normalized [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global Average Pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def generate_heatmap_overlay(self, input_tensor, original_image, class_idx=None, alpha=0.4):
        """
        Generate colored heatmap overlaid on original image.

        Args:
            input_tensor: Preprocessed tensor [1, 3, 224, 224]
            original_image: Original image as numpy array [H, W, 3] (BGR)
            class_idx: Target class index
            alpha: Overlay transparency

        Returns:
            overlay: Heatmap overlaid image [H, W, 3]
        """
        # Generate CAM
        cam = self.generate(input_tensor, class_idx)

        # Resize to original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap (JET: blue→green→red)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )

        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

        return overlay, cam_resized

    def save_heatmap(self, input_tensor, original_image, class_idx, output_path, alpha=0.4):
        """Generate and save heatmap to file."""
        overlay, cam = self.generate_heatmap_overlay(
            input_tensor, original_image, class_idx, alpha
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)

        return str(output_path)
