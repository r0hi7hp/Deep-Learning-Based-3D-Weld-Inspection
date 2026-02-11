"""
Training script for KPConv Segmentation Model
Optimized for weld detection with advanced training strategies
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

import matplotlib
matplotlib.use('Agg')  # Prevent UI hang
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from kpconv_model import KPConvSegmentation, CombinedLoss
from weld_dataset import KPConvWeldDataset

# Minimal logging
logger = logging.getLogger(__name__)


class Trainer:
    """KPConv training class with comprehensive metrics tracking"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create model (simplified API)
        self.model = KPConvSegmentation(
            num_classes=config['num_classes'],
            in_channels=config.get('in_channels', 3),
            init_features=config.get('init_features', 64),
            k=config.get('k_neighbors', 16)
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model has {total_params:,} parameters")
        
        # Setup loss function
        if config.get('use_simple_loss', True):
            class_weights = torch.tensor(
                [1.0, config['weld_weight']], 
                dtype=torch.float32
            ).to(self.device)
            self.criterion = nn.NLLLoss(weight=class_weights)
            logger.info(f"Using NLLLoss with weld weight: {config['weld_weight']}")
        else:
            self.criterion = CombinedLoss().to(self.device)
            logger.info("Using CombinedLoss (Focal + Dice)")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < config['warmup_epochs']:
                return (epoch + 1) / config['warmup_epochs']
            else:
                progress = (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
                return max(config['min_lr'] / config['learning_rate'],
                          0.5 * (1 + np.cos(np.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_ious = []
        self.val_ious = []
        self.learning_rates = []
        
        self.best_val_iou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create output directories
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints'
        self.metrics_dir = Path(__file__).parent / 'metrics'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
    
    def calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """Calculate accuracy and IoU"""
        pred = torch.argmax(logits, dim=1)
        
        correct = (pred == labels).sum().item()
        total = labels.numel()
        accuracy = correct / total
        
        pred_weld = (pred == 1)
        true_weld = (labels == 1)
        
        intersection = (pred_weld & true_weld).sum().item()
        union = (pred_weld | true_weld).sum().item()
        iou = intersection / union if union > 0 else 0.0
        
        return accuracy, iou
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_iou = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (points, labels) in enumerate(progress_bar):
            points = points.to(self.device)
            labels = labels.to(self.device)
            points = points.transpose(1, 2)
            
            self.optimizer.zero_grad()
            logits = self.model(points)
            loss = self.criterion(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            with torch.no_grad():
                acc, iou = self.calculate_metrics(logits, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_iou += iou
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}', 'iou': f'{iou:.4f}'})
        
        if num_batches > 0:
            return epoch_loss / num_batches, epoch_acc / num_batches, epoch_iou / num_batches
        return 0.0, 0.0, 0.0
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0.0
        val_acc = 0.0
        val_iou = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for points, labels in tqdm(val_loader, desc="Validating"):
                points = points.to(self.device).transpose(1, 2)
                labels = labels.to(self.device)
                
                logits = self.model(points)
                loss = self.criterion(logits, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    acc, iou = self.calculate_metrics(logits, labels)
                    val_loss += loss.item()
                    val_acc += acc
                    val_iou += iou
                    num_batches += 1
        
        if num_batches > 0:
            return val_loss / num_batches, val_acc / num_batches, val_iou / num_batches
        return 0.0, 0.0, 0.0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_iou': self.best_val_iou,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest_model.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            logger.info(f"Saved best model with IoU: {self.best_val_iou:.4f}")
        
        if (epoch + 1) % self.config['save_interval'] == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'model_epoch_{epoch+1}.pth')
    
    def plot_metrics(self):
        """Plot training metrics with enhanced visualization"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # 1. Training Metrics Plot (Loss, Accuracy, IoU, LR)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        axes[1, 0].plot(epochs, self.train_ious, 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.val_ious, 'r-', label='Val', linewidth=2)
        axes[1, 0].set_title('IoU (Weld Class)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'training_metrics.png', dpi=300)
        plt.close()
        logger.info(f"Saved training metrics to {self.metrics_dir / 'training_metrics.png'}")
        
        # 2. IoU Score Plot (dedicated)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.train_ious, 'b-', label='Training Weld IoU', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, self.val_ious, 'r-', label='Validation Weld IoU', linewidth=2, marker='s', markersize=3)
        ax.set_title('IoU Score Over Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('IoU Score', fontsize=12)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Annotate best epoch
        if self.val_ious:
            best_epoch_idx = np.argmax(self.val_ious)
            best_epoch_num = best_epoch_idx + 1
            ax.axvline(x=best_epoch_num, color='green', linestyle=':', alpha=0.7)
            ax.annotate(f'Best: {self.best_val_iou:.4f}', 
                       xy=(best_epoch_num, self.best_val_iou),
                       xytext=(best_epoch_num + 5, self.best_val_iou - 0.1),
                       fontsize=10,
                       arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'iou_scores.png', dpi=150)
        plt.close()
        logger.info(f"Saved IoU scores to {self.metrics_dir / 'iou_scores.png'}")
    
    def save_metrics(self):
        """Save metrics to JSON"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'learning_rates': self.learning_rates,
            'best_val_iou': self.best_val_iou,
            'best_epoch': self.best_epoch
        }
        
        with open(self.metrics_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc, train_iou = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_iou = self.validate(val_loader, epoch)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_ious.append(train_iou)
            self.val_ious.append(val_iou)
            self.learning_rates.append(current_lr)
            
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, IoU: {train_iou:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, IoU: {val_iou:.4f}")
            
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                self.best_epoch = epoch
                self.patience_counter = 0
                logger.info(f"  New best IoU: {val_iou:.4f}!")
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            if self.patience_counter >= self.config['patience']:
                logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        self.plot_metrics()
        self.save_metrics()
        self._generate_confusion_matrix(val_loader)
        logger.info(f"\nTraining complete! Best IoU: {self.best_val_iou:.4f}")
    
    def _generate_confusion_matrix(self, val_loader: DataLoader):
        """Generate confusion matrix and training summary plot"""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(self.device).transpose(1, 2)
                outputs = self.model(points)
                preds = outputs.max(dim=1)[1]
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
        
        cm = confusion_matrix(all_labels, all_preds)
        
        # 1. Standalone confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Background', 'Weld'])
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        ax.set_title('Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.metrics_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix to {self.metrics_dir / 'confusion_matrix.png'}")
        
        # 2. Combined training summary plot
        epochs = range(1, len(self.train_losses) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Weld IoU
        axes[1, 0].plot(epochs, self.train_ious, 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.val_ious, 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('Weld IoU', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Confusion matrix in bottom right
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BG', 'Weld'])
        disp2.plot(cmap=plt.cm.Blues, ax=axes[1, 1], values_format='d')
        axes[1, 1].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Calculate best validation accuracy
        best_val_acc = max(self.val_accs) if self.val_accs else 0
        
        plt.suptitle(f'KPConv Training Summary\nBest Val Weld IoU: {self.best_val_iou:.4f} | Best Val Acc: {best_val_acc:.4f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.metrics_dir / 'training_summary.png', dpi=150)
        plt.close()
        logger.info(f"Saved training summary to {self.metrics_dir / 'training_summary.png'}")


def main():
    """Main training function"""
    config = {
        # Model (optimized for speed)
        'num_classes': 2,
        'in_channels': 3,
        'init_features': 32,   # Reduced from 64
        'k_neighbors': 12,     # Reduced from 16
        
        # Data (optimized)
        'dataset_path': r'C:\Users\rohith.p\Desktop\Weld Inspection\Weld Path\Dataset',
        'num_points': 1024,    # Reduced from 2048 for ~2x speedup
        'batch_size': 8,       # Increased from 4 (faster if GPU memory allows)
        'num_workers': 4,
        'num_train_models': 70,
        'num_val_models': 30,
        
        # Training
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # Loss
        'weld_weight': 20.0,
        'use_simple_loss': True,
        
        # Scheduler
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        
        # Checkpointing
        'save_interval': 10,
        'patience': 15,
    }
    
    # Load dataset files
    dataset_path = Path(config['dataset_path'])
    all_files = []
    for model_dir in sorted(dataset_path.glob('model_*')):
        if model_dir.is_dir():
            npz_files = list(model_dir.glob('label_*.npz'))
            if len(npz_files) == 1:
                all_files.append(npz_files[0])
    
    logger.info(f"Found {len(all_files)} total models")
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_files)
    
    train_files = all_files[:config['num_train_models']]
    val_files = all_files[config['num_train_models']:config['num_train_models'] + config['num_val_models']]
    
    logger.info(f"Training: {len(train_files)}, Validation: {len(val_files)}")
    
    # Create datasets and loaders
    train_dataset = KPConvWeldDataset(train_files, config['num_points'], 'all', True)
    val_dataset = KPConvWeldDataset(val_files, config['num_points'], 'all', False)
    
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, 
                             num_workers=config['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True)
    
    # Train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
