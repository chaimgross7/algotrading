"""LSTM model for time-series prediction."""

from typing import Dict
import torch
import torch.nn as nn

from algotrade.models.base_model import BaseModel


class LSTMModel(BaseModel):
    """Multi-layer LSTM with prediction heads for direction/magnitude/volatility."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__(input_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        backbone_dim = hidden_dim * self.num_directions
        self.layer_norm = nn.LayerNorm(backbone_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_heads(backbone_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                         device=x.device, dtype=x.dtype)
        c0 = torch.zeros_like(h0)
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        
        # Use last layer's hidden state
        if self.num_directions == 2:
            final = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            final = hn[-1]
        
        final = self.dropout(self.layer_norm(final))
        return self._predict_from_backbone(final)
