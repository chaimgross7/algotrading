"""Training module for supervised and RL training."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, SequentialLR
from tqdm import tqdm
import logging

from algotrade.models.losses import MultiTaskLoss, UncertaintyWeightedLoss

logger = logging.getLogger("algotrade.training")


@dataclass
class TrainerConfig:
    """Configuration for training."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    class_weights: Optional[List[float]] = None
    use_uncertainty_loss: bool = False
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    # Loss weights
    direction_weight: float = 1.0
    magnitude_weight: float = 0.5
    volatility_weight: float = 0.3
    # Scheduler parameters
    scheduler_patience: int = 10
    scheduler_factor: float = 0.1
    scheduler_max_reductions: int = 4
    scheduler_restart_lr_factor: float = 0.5
    
    @classmethod
    def from_config(cls, train_cfg: Dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from config dict."""
        loss_weights = train_cfg.get("loss_weights", {})
        scheduler_cfg = train_cfg.get("scheduler", {})
        
        return cls(
            lr=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
            use_uncertainty_loss=train_cfg.get("use_uncertainty", False),
            warmup_epochs=train_cfg.get("warmup_epochs", 5),
            label_smoothing=train_cfg.get("label_smoothing", 0.1),
            direction_weight=train_cfg.get("direction_weight", loss_weights.get("direction", 1.0)),
            magnitude_weight=train_cfg.get("magnitude_weight", loss_weights.get("magnitude", 0.5)),
            volatility_weight=train_cfg.get("volatility_weight", loss_weights.get("volatility", 0.3)),
            scheduler_patience=scheduler_cfg.get("patience", 10),
            scheduler_factor=scheduler_cfg.get("factor", 0.1),
            scheduler_max_reductions=scheduler_cfg.get("max_reductions", 4),
            scheduler_restart_lr_factor=scheduler_cfg.get("restart_lr_factor", 0.5),
        )


class AdaptivePlateauScheduler:
    """LR scheduler that reduces on plateau and restarts after multiple reductions to escape local minima.
    
    Reduces LR by `factor` (default 0.1) when validation loss plateaus.
    After `max_reductions` consecutive reductions, restarts LR to `initial_lr * restart_lr_factor`
    to escape potential local minima. The reduction counter resets after each restart.
    """
    
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        max_reductions: int = 4,
        restart_lr_factor: float = 0.5,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.factor = factor
        self.max_reductions = max_reductions
        self.restart_lr_factor = restart_lr_factor
        self.min_lr = min_lr
        self.reduction_count = 0
        
        # Underlying ReduceLROnPlateau scheduler
        self._base_scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )
        
        # Track current LR to detect reductions
        self._last_lr = self._get_lr()
    
    def _get_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]['lr']
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self, val_loss: float):
        """Step the scheduler with validation loss.
        
        Detects LR reductions, tracks count, and triggers restart after max_reductions.
        """
        # Let base scheduler decide whether to reduce
        self._base_scheduler.step(val_loss)
        
        current_lr = self._get_lr()
        
        # Check if LR was reduced
        if current_lr < self._last_lr:
            self.reduction_count += 1
            logger.info(
                f"LR reduced: {self._last_lr:.2e} -> {current_lr:.2e} "
                f"(reduction {self.reduction_count}/{self.max_reductions})"
            )
            
            # Check if we should restart to escape local minima
            if self.reduction_count >= self.max_reductions:
                restart_lr = self.initial_lr * self.restart_lr_factor
                logger.info(
                    f"LR RESTART: {current_lr:.2e} -> {restart_lr:.2e} "
                    f"(50% of initial LR to escape local minima)"
                )
                self._set_lr(restart_lr)
                self.reduction_count = 0
                
                # Reset base scheduler's internal state for the new LR
                self._base_scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode=self._base_scheduler.mode,
                    factor=self.factor,
                    patience=self._base_scheduler.patience,
                    min_lr=self.min_lr,
                )
                current_lr = restart_lr
        
        self._last_lr = current_lr
    
    def get_last_lr(self) -> List[float]:
        """Get last computed learning rate."""
        return [self._get_lr()]


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> List[float]:
    """Compute inverse frequency class weights for imbalanced labels."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # Inverse frequency with smoothing
    weights = total / (num_classes * counts + 1e-6)
    # Normalize so mean weight is 1
    weights = weights / weights.mean()
    return weights.tolist()


class Trainer:
    """Supervised trainer for multi-task prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        config: TrainerConfig = None,
        **kwargs,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            device: Torch device (auto-detect if None)
            config: TrainerConfig object (preferred)
            **kwargs: Individual config params (for backward compatibility)
        """
        # Support both config object and individual kwargs
        if config is None:
            config = TrainerConfig(**{k: v for k, v in kwargs.items() if k in TrainerConfig.__dataclass_fields__})
        
        self.config = config
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Select loss function
        if config.use_uncertainty_loss:
            self.criterion = UncertaintyWeightedLoss(label_smoothing=config.label_smoothing)
            logger.info("Using UncertaintyWeightedLoss (auto-learned task weights)")
        else:
            self.criterion = MultiTaskLoss(
                direction_weight=config.direction_weight,
                magnitude_weight=config.magnitude_weight,
                volatility_weight=config.volatility_weight,
                class_weights=config.class_weights,
                label_smoothing=config.label_smoothing,
            )
            if config.class_weights is not None:
                logger.info(f"Using MultiTaskLoss with class weights: {config.class_weights}")
            if config.direction_weight == 0:
                logger.info("Direction loss disabled - training for magnitude regression only")
        
        self.criterion = self.criterion.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # LR scheduling with warmup and adaptive plateau with restart
        self.warmup_epochs = config.warmup_epochs
        self.scheduler = None  # Created in train() when we know total epochs
        self._base_scheduler = AdaptivePlateauScheduler(
            self.optimizer,
            initial_lr=config.lr,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            max_reductions=config.scheduler_max_reductions,
            restart_lr_factor=config.scheduler_restart_lr_factor,
        )
        logger.info(
            f"Using AdaptivePlateauScheduler: factor={config.scheduler_factor}, patience={config.scheduler_patience}, "
            f"max_reductions={config.scheduler_max_reductions}, restart_factor={config.scheduler_restart_lr_factor}"
        )
        
        self.best_loss = float("inf")
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: Dict[str, torch.Tensor],
        X_val: torch.Tensor = None,
        y_val: Dict[str, torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 20,
    ) -> Dict[str, list]:
        """Train the model."""
        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size) if X_val is not None else None
        
        # Create warmup + plateau scheduler
        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / max(1, self.warmup_epochs))
        )
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        no_improve = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                # Use warmup for early epochs, then plateau scheduler
                if epoch < self.warmup_epochs:
                    warmup_scheduler.step()
                else:
                    self._base_scheduler.step(val_loss)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save("best.pt")
                    no_improve = 0
                else:
                    no_improve += 1
            
            # Log uncertainty weights if using UncertaintyWeightedLoss
            extra_info = ""
            if isinstance(self.criterion, UncertaintyWeightedLoss):
                weights = self.criterion.get_weights()
                extra_info = f" - task_weights: dir={weights['direction']:.2f}, mag={weights['magnitude']:.2f}, vol={weights['volatility']:.2f}"
            
            logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.2%}{extra_info}")
            
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for X, y_dir, y_mag, y_vol in tqdm(loader, desc="Training", leave=False):
            X = X.to(self.device)
            targets = {
                "direction": y_dir.to(self.device),
                "magnitude": y_mag.to(self.device),
                "volatility": y_vol.to(self.device),
            }
            
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for X, y_dir, y_mag, y_vol in loader:
                X = X.to(self.device)
                targets = {
                    "direction": y_dir.to(self.device),
                    "magnitude": y_mag.to(self.device),
                    "volatility": y_vol.to(self.device),
                }
                
                preds = self.model(X)
                loss = self.criterion(preds, targets)
                total_loss += loss.item()
                
                pred_dir = preds["direction"].argmax(dim=-1)
                correct += (pred_dir == targets["direction"]).sum().item()
                total += X.size(0)
        
        return total_loss / len(loader), correct / total if total > 0 else 0.0
    
    def _make_loader(self, X: torch.Tensor, y: Dict[str, torch.Tensor], batch_size: int, shuffle: bool = False) -> DataLoader:
        if X is None:
            return None
        dataset = TensorDataset(X, y["direction"], y["magnitude"], y["volatility"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def save(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }, path)
    
    def load(self, filename: str):
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
