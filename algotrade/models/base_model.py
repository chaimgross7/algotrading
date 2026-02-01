"""Base model class for multi-task prediction."""

from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base for all prediction models. Outputs direction, magnitude, volatility."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dict with 'direction' (logits), 'magnitude', 'volatility'."""
        pass
    
    def _init_heads(self, backbone_dim: int):
        """Initialize prediction heads."""
        self.direction_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(backbone_dim // 2, 3),  # up/flat/down
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(backbone_dim // 2, 1),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(backbone_dim // 2, 1),
        )
    
    def _predict_from_backbone(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions from backbone output."""
        return {
            "direction": self.direction_head(h),
            "magnitude": self.magnitude_head(h),
            "volatility": self.volatility_head(h),
        }
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
