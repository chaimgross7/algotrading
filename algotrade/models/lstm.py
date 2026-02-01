"""LSTM model for time-series prediction."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from algotrade.models.base_model import BaseModel


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence dimension."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling.
        
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            pooled: (batch, hidden_dim)
        """
        # Compute attention scores
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class LSTMModel(BaseModel):
    """Multi-layer LSTM with attention pooling and prediction heads for direction/magnitude/volatility."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention_pooling: bool = True,
    ):
        super().__init__(input_dim, hidden_dim, dropout)
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.use_attention_pooling = use_attention_pooling
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        backbone_dim = hidden_dim * self.num_directions
        
        # Attention pooling over all timesteps
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(backbone_dim)
        
        self.layer_norm = nn.LayerNorm(backbone_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_heads(backbone_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                         device=x.device, dtype=x.dtype)
        c0 = torch.zeros_like(h0)
        
        # Get all hidden states
        output, (hn, _) = self.lstm(x, (h0, c0))  # output: (batch, seq_len, hidden*directions)
        
        if self.use_attention_pooling:
            # Attention-weighted pooling over all timesteps
            final = self.attention_pool(output)
        else:
            # Fallback: use last layer's hidden state
            if self.num_directions == 2:
                final = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                final = hn[-1]
        
        final = self.dropout(self.layer_norm(final))
        return self._predict_from_backbone(final)
