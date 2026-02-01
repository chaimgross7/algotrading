"""Training module for supervised and RL training."""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging

from algotrade.models.losses import MultiTaskLoss

logger = logging.getLogger("algotrade.training")


class Trainer:
    """Supervised trainer for multi-task prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
        class_weights: list = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = MultiTaskLoss(class_weights=class_weights)
        if class_weights is not None:
            self.criterion.ce = self.criterion.ce.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)
        
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
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save("best.pt")
                    no_improve = 0
                else:
                    no_improve += 1
            
            logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.2%}")
            
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
