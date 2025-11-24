"""
Model architecture for land use classification using transfer learning.
Uses ResNet-50 pre-trained on ImageNet with custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class LandUseClassifier(nn.Module):
    """
    CNN model for binary land use classification (City vs Farmland).
    Based on ResNet-50 with transfer learning.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            num_classes: Number of output classes (default: 2 for binary)
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: If True, freeze early convolutional layers
            dropout_rate: Dropout rate in classification head
        """
        super(LandUseClassifier, self).__init__()
        
        # Load pre-trained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Freeze early layers if specified
        if freeze_backbone:
            # Freeze all layers except the last residual block (layer4) and FC
            for name, param in self.backbone.named_parameters():
                if not name.startswith('layer4') and not name.startswith('fc'):
                    param.requires_grad = False
        
        # Get the number of features from the original FC layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with custom classification head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_trainable_params(self):
        """Returns number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Returns total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def get_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
    device: str = 'auto'
) -> LandUseClassifier:
    """
    Factory function to create and initialize the model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze early convolutional layers
        dropout_rate: Dropout rate in classification head
        device: Device to move model to ('auto', 'cuda', 'mps', or 'cpu')
    
    Returns:
        Initialized model on specified device
    """
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Creating LandUseClassifier model...")
    print(f"  Pre-trained: {pretrained}")
    print(f"  Frozen backbone: {freeze_backbone}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Device: {device}")
    
    model = LandUseClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {model.get_total_params():,}")
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    print(f"  Frozen parameters: {model.get_total_params() - model.get_trainable_params():,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...\n")
    
    model = get_model(device='cpu')
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output}")
