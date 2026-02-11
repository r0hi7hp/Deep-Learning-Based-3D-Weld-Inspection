"""
PointNet Training Script for Weld Detection - Production Ready
Optimized for deployment with memory management and clean output.
"""
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from weld_dataset import WeldDataset
from pointnet import PointNetSemSeg
from pathlib import Path
import numpy as np
import os
import random
import json
from typing import Dict, List, Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_model_state: Optional[Dict] = None
        
    def __call__(self, val_score: float, model: nn.Module) -> None:
        if self.best_score is None:
            self.best_score = val_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            
    def load_best_model(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """Regularization loss for feature transform matrix."""
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def calculate_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> List[float]:
    """Calculate IoU for each class."""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        iou = 1.0 if union == 0 else intersection / union
        ious.append(float(iou))
    
    return ious


class PointNetTrainer:
    """Trainer class for PointNet weld detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        self.setup_model()
        
    def setup_directories(self) -> None:
        """Setup checkpoint and metrics directories."""
        script_dir = Path(__file__).parent.resolve()
        self.save_dir = script_dir / 'checkpoints'
        self.metrics_dir = script_dir / 'Metrics'
        
        # Get next training ID
        next_id = self._get_next_training_id()
        folder_name = f"Training {next_id}"
        
        self.save_dir = self.save_dir / folder_name
        self.metrics_dir = self.metrics_dir / folder_name
        
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
    def _get_next_training_id(self) -> int:
        max_id = 0
        for base_dir in [self.save_dir, self.metrics_dir]:
            base_dir.mkdir(exist_ok=True, parents=True)
            for d in base_dir.iterdir():
                if d.is_dir() and d.name.startswith("Training"):
                    parts = d.name.split(' ')
                    if len(parts) > 1 and parts[-1].isdigit():
                        max_id = max(max_id, int(parts[-1]))
        return max_id + 1
        
    def setup_model(self) -> None:
        """Initialize model, optimizer, and criterion."""
        self.model = PointNetSemSeg(num_classes=2).to(self.device)
        
        class_weights = torch.tensor([1.0, self.config['weld_weight']]).to(self.device)
        self.criterion = nn.NLLLoss(weight=class_weights)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['epochs'], eta_min=1e-5
        )
        
    def prepare_data(self, data_dir: Path) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        original_files = []
        for model_dir in sorted(data_dir.glob('model_*')):
            if model_dir.is_dir():
                npz_files = list(model_dir.glob('label_*.npz'))
                if len(npz_files) == 1:
                    original_files.append(npz_files[0])
        
        # Shuffle and split 70/30
        random.seed(42)
        random.shuffle(original_files)
        
        split_idx = int(len(original_files) * 0.7)
        train_files = original_files[:split_idx]
        val_files = original_files[split_idx:]
        
        # Save split info
        split_info = {
            'train_files': [str(f) for f in train_files],
            'val_files': [str(f) for f in val_files],
            'total_files': len(original_files)
        }
        with open(self.metrics_dir / 'data_split.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        train_dataset = WeldDataset(file_list=train_files, num_points=self.config['num_points'], split='all')
        val_dataset = WeldDataset(file_list=val_files, num_points=self.config['num_points'], split='all')
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        
        return train_loader, val_loader
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        epoch_preds, epoch_labels = [], []
        
        for points, labels in train_loader:
            points = points.permute(0, 2, 1).to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(points)
            loss = self.criterion(pred, labels)
            
            if self.config['feature_reg'] > 0:
                _, trans_feat = self.model.get_transforms(points)
                if trans_feat is not None:
                    loss += self.config['feature_reg'] * feature_transform_regularizer(trans_feat)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct_pixels += pred_choice.eq(labels.data).cpu().sum().item()
            total_pixels += labels.numel()
            
            epoch_preds.append(pred_choice.cpu().numpy())
            epoch_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_pixels / total_pixels
        
        all_preds = np.concatenate(epoch_preds).flatten()
        all_labels = np.concatenate(epoch_labels).flatten()
        weld_iou = calculate_iou(all_preds, all_labels)[1]
        
        # Memory cleanup
        del epoch_preds, epoch_labels, all_preds, all_labels
        
        return avg_loss, accuracy, weld_iou
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Run validation."""
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds, val_labels_list = [], []
        
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.permute(0, 2, 1).to(self.device)
                labels = labels.to(self.device)
                
                pred = self.model(points)
                loss = self.criterion(pred, labels)
                val_loss += loss.item()
                
                pred_choice = pred.data.max(1)[1]
                val_correct += pred_choice.eq(labels.data).cpu().sum().item()
                val_total += labels.numel()
                
                val_preds.append(pred_choice.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
        
        avg_loss = val_loss / len(val_loader)
        accuracy = val_correct / val_total
        
        all_preds = np.concatenate(val_preds).flatten()
        all_labels = np.concatenate(val_labels_list).flatten()
        weld_iou = calculate_iou(all_preds, all_labels)[1]
        
        # Memory cleanup
        del val_preds, val_labels_list
        
        return avg_loss, accuracy, weld_iou
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Run full training loop."""
        early_stopping = EarlyStopping(patience=self.config['patience'])
        best_val_iou = 0.0
        
        metrics = {
            'train_losses': [], 'val_losses': [],
            'train_accuracies': [], 'val_accuracies': [],
            'train_weld_ious': [], 'val_weld_ious': []
        }
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc, train_iou = self.train_epoch(train_loader)
            val_loss, val_acc, val_iou = self.validate(val_loader)
            
            metrics['train_losses'].append(train_loss)
            metrics['val_losses'].append(val_loss)
            metrics['train_accuracies'].append(train_acc)
            metrics['val_accuracies'].append(val_acc)
            metrics['train_weld_ious'].append(train_iou)
            metrics['val_weld_ious'].append(val_iou)
            
            print(f"Epoch [{epoch+1}/{self.config['epochs']}] "
                  f"Train: Loss={train_loss:.4f}, IoU={train_iou:.4f} | "
                  f"Val: Loss={val_loss:.4f}, IoU={val_iou:.4f}")
            
            self.scheduler.step()
            
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
            
            early_stopping(val_iou, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        torch.save(self.model.state_dict(), self.save_dir / "final_model.pth")
        
        # Save metrics
        with open(self.metrics_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
        
    def cleanup(self) -> None:
        """Release resources."""
        del self.model, self.optimizer, self.scheduler
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PointNet for weld detection')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weld-weight', type=float, default=20.0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--feature-reg', type=float, default=0.001)
    args = parser.parse_args()
    
    config = {
        'batch_size': args.batch_size,
        'num_points': 2048,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-3,
        'weld_weight': args.weld_weight,
        'patience': args.patience,
        'feature_reg': args.feature_reg
    }
    
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir.parent / 'Dataset'
    
    trainer = PointNetTrainer(config)
    train_loader, val_loader = trainer.prepare_data(data_dir)
    trainer.train(train_loader, val_loader)
    trainer.cleanup()
    
    print("Training complete!")


if __name__ == '__main__':
    main()
