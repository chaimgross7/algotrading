"""Loss functions for multi-task learning."""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Combined loss for direction (CE) + magnitude (MSE) + volatility (MSE)."""
    
    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 1.0,
        volatility_weight: float = 0.5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.weights = {"direction": direction_weight, "magnitude": magnitude_weight, "volatility": volatility_weight}
        # Class weights: [down, flat, up] - higher weight for minority classes
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        total = 0.0
        
        if "direction" in preds and "direction" in targets:
            losses["direction"] = self.ce(preds["direction"], targets["direction"].long())
            total += self.weights["direction"] * losses["direction"]
        
        if "magnitude" in preds and "magnitude" in targets:
            t = targets["magnitude"].unsqueeze(-1) if targets["magnitude"].dim() == 1 else targets["magnitude"]
            losses["magnitude"] = self.mse(preds["magnitude"], t)
            total += self.weights["magnitude"] * losses["magnitude"]
        
        if "volatility" in preds and "volatility" in targets:
            t = targets["volatility"].unsqueeze(-1) if targets["volatility"].dim() == 1 else targets["volatility"]
            losses["volatility"] = self.mse(preds["volatility"], t)
            total += self.weights["volatility"] * losses["volatility"]
        
        return (total, losses) if return_components else total


class UncertaintyWeightedLoss(nn.Module):
    """Learns task weights via homoscedastic uncertainty (Kendall et al., 2018)."""
    
    def __init__(self):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            "direction": nn.Parameter(torch.zeros(1)),
            "magnitude": nn.Parameter(torch.zeros(1)),
            "volatility": nn.Parameter(torch.zeros(1)),
        })
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        total = 0.0
        
        if "direction" in preds and "direction" in targets:
            loss = self.ce(preds["direction"], targets["direction"].long())
            precision = torch.exp(-self.log_vars["direction"])
            losses["direction"] = loss
            total += precision * loss + self.log_vars["direction"]
        
        for key in ["magnitude", "volatility"]:
            if key in preds and key in targets:
                t = targets[key].unsqueeze(-1) if targets[key].dim() == 1 else targets[key]
                loss = self.mse(preds[key], t)
                precision = torch.exp(-self.log_vars[key])
                losses[key] = loss
                total += 0.5 * precision * loss + 0.5 * self.log_vars[key]
        
        return (total, losses) if return_components else total
    
    def get_weights(self) -> Dict[str, float]:
        return {k: torch.exp(-v).item() for k, v in self.log_vars.items()}
