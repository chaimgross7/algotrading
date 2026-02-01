"""Model factory for creating prediction models."""

from typing import Dict, Any
import torch

from algotrade.models.base_model import BaseModel
from algotrade.models.lstm import LSTMModel
from algotrade.models.transformer import TransformerModel


def create_model(input_dim: int, model_cfg: Dict[str, Any]) -> BaseModel:
    """Create a model based on configuration.
    
    Args:
        input_dim: Number of input features
        model_cfg: Model configuration dict with 'type' and model-specific params
        
    Returns:
        Instantiated model (LSTMModel or TransformerModel)
    """
    model_type = model_cfg.get("type", "lstm")
    hidden_dim = model_cfg.get("hidden_dim", 128)
    dropout = model_cfg.get("dropout", 0.2)
    
    if model_type == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=model_cfg.get("num_layers", 2),
            dropout=dropout,
            bidirectional=model_cfg.get("bidirectional", False),
            use_attention_pooling=model_cfg.get("use_attention_pooling", True),
        )
    elif model_type == "transformer":
        transformer_cfg = model_cfg.get("transformer", {})
        return TransformerModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=transformer_cfg.get("num_heads", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            ff_dim=transformer_cfg.get("ff_dim", 512),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'lstm', 'transformer'")


def load_model(
    checkpoint_path: str,
    input_dim: int,
    model_cfg: Dict[str, Any],
    device: torch.device = None,
) -> BaseModel:
    """Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_dim: Number of input features
        model_cfg: Model configuration dict
        device: Device to load model on (auto-detect if None)
        
    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(input_dim, model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model
