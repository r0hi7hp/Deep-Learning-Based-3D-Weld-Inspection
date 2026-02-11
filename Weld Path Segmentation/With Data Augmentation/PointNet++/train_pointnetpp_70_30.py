"""
PointNet++ Training Script for Weld Detection (with Rotation Augmentation)
- Uses combined dataset (original + augmented) with 70:30 train/val split
- All data (original + augmented) treated equally since augmentation is pure rotation
- 150 epochs with early stopping (patience 15)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from weld_dataset import WeldDataset
from pointnet2 import PointNet2SemSeg


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_score, model):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def calculate_iou(pred, target, num_classes=2):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)
    
    return ious


def calculate_metrics(pred, target, num_classes=2):
    """Calculate precision, recall, f1 for each class"""
    metrics = {}
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        tp = (pred_cls & target_cls).sum()
        fp = (pred_cls & ~target_cls).sum()
        fn = (~pred_cls & target_cls).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = 'background' if cls == 0 else 'weld'
        metrics[f'{class_name}_precision'] = precision
        metrics[f'{class_name}_recall'] = recall
        metrics[f'{class_name}_f1'] = f1
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PointNet++ for weld detection (70:30 split)')
    parser.add_argument('--weld-weight', type=float, default=20.0,
                       help='Class weight for weld class (default: 20.0)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs (default: 150)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Hyperparameters
    BATCH_SIZE = args.batch_size
    NUM_POINTS = 2048
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = 1e-3
    WELD_WEIGHT = args.weld_weight
    EARLY_STOPPING_PATIENCE = args.patience
    
    # Paths
    ROOT_DIR = SCRIPT_DIR.parent.parent / 'Dataset'
    BASE_SAVE_DIR = SCRIPT_DIR / 'checkpoints'
    BASE_METRICS_DIR = SCRIPT_DIR / 'Metrics'
    
    logging.info("=" * 60)
    logging.info("PointNet++ Training (70:30 Split with Augmented Data)")
    logging.info("=" * 60)
    logging.info(f"Dataset directory: {ROOT_DIR}")
    
    # Helper for incremental folders
    def get_next_training_id(base_dirs, prefix="Training"):
        max_id = 0
        for base_dir in base_dirs:
            Path(base_dir).mkdir(exist_ok=True, parents=True)
            existing_dirs = [d.name for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith(prefix)]
            for d in existing_dirs:
                parts = d.split(' ')
                if len(parts) > 1 and parts[-1].isdigit():
                    max_id = max(max_id, int(parts[-1]))
        return max_id + 1

    next_id = get_next_training_id([str(BASE_SAVE_DIR), str(BASE_METRICS_DIR)])
    folder_name = f"Training {next_id}"
    
    SAVE_DIR = BASE_SAVE_DIR / folder_name
    METRICS_DIR = BASE_METRICS_DIR / folder_name
    
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Saving checkpoints to: {SAVE_DIR}")
    logging.info(f"Saving metrics to: {METRICS_DIR}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # =========================================================================
    # Dataset Preparation - 70:30 split on combined data
    # =========================================================================
    root_path = Path(ROOT_DIR)
    augmented_data_path = root_path / 'Augmented_Rotation_Data'
    
    # Load original files
    original_files = []
    for model_dir in sorted(root_path.glob('model_*')):
        if not model_dir.is_dir():
            continue
        npz_files = list(model_dir.glob('label_*.npz'))
        if len(npz_files) == 1:
            original_files.append(npz_files[0])
            
    logging.info(f"Found {len(original_files)} original models")
    
    # Load augmented files
    augmented_files = []
    if augmented_data_path.exists():
        augmented_files = list(augmented_data_path.glob('*.npz'))
    logging.info(f"Found {len(augmented_files)} augmented files")
    
    # Combine all files and shuffle
    all_files = original_files + augmented_files
    random.seed(42)
    random.shuffle(all_files)
    
    # 70:30 split
    train_count = int(len(all_files) * 0.7)
    train_files = all_files[:train_count]
    val_files = all_files[train_count:]
    
    logging.info("=" * 50)
    logging.info("DATA SPLIT (70:30 on Combined Data)")
    logging.info("=" * 50)
    logging.info(f"Total files: {len(all_files)} ({len(original_files)} original + {len(augmented_files)} augmented)")
    logging.info(f"Training: {len(train_files)} files (70%)")
    logging.info(f"Validation: {len(val_files)} files (30%)")
    logging.info("=" * 50)
    
    # Save split info
    split_info = {
        'train_files': [str(f) for f in train_files],
        'val_files': [str(f) for f in val_files],
        'random_seed': 42,
        'total_files': len(all_files),
        'original_count': len(original_files),
        'augmented_count': len(augmented_files),
        'train_count': len(train_files),
        'val_count': len(val_files)
    }
    with open(METRICS_DIR / 'data_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create datasets - NO online augmentation since we use offline augmented data
    train_dataset = WeldDataset(file_list=train_files, num_points=NUM_POINTS, split='all', augment=False)
    val_dataset = WeldDataset(file_list=val_files, num_points=NUM_POINTS, split='all', augment=False)
    
    logging.info(f"Training on {len(train_dataset)} samples")
    logging.info(f"Validating on {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = PointNet2SemSeg(num_classes=2).to(device)
    
    # Loss and Optimizer
    class_weights = torch.tensor([1.0, WELD_WEIGHT]).to(device)
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    
    best_val_iou = 0.0
    best_acc = 0.0
    
    # Metrics storage
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_ious = []
    val_ious = []
    train_weld_ious = []
    val_weld_ious = []
    learning_rates = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        epoch_preds = []
        epoch_labels = []
        
        for i, (points, labels) in enumerate(train_loader):
            points = points.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pred = outputs.max(dim=1)[1]
            correct_pixels += pred.eq(labels).sum().item()
            total_pixels += labels.numel()
            
            epoch_preds.append(pred.cpu().numpy())
            epoch_labels.append(labels.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_pixels / total_pixels
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # Calculate training IoU
        epoch_preds_all = np.concatenate(epoch_preds).flatten()
        epoch_labels_all = np.concatenate(epoch_labels).flatten()
        train_epoch_ious = calculate_iou(epoch_preds_all, epoch_labels_all)
        train_ious.append(train_epoch_ious)
        train_weld_ious.append(train_epoch_ious[1])
        
        # Validation Loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.permute(0, 2, 1).to(device)
                labels = labels.to(device)
                
                outputs = model(points)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                pred = outputs.max(dim=1)[1]
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.numel()
                
                val_preds.append(pred.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Calculate validation IoU
        val_preds_all = np.concatenate(val_preds).flatten()
        val_labels_all = np.concatenate(val_labels_list).flatten()
        val_epoch_ious = calculate_iou(val_preds_all, val_labels_all)
        val_ious.append(val_epoch_ious)
        val_weld_ious.append(val_epoch_ious[1])
        
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}] - "
                    f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}, Train Weld IoU: {train_epoch_ious[1]:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Weld IoU: {val_epoch_ious[1]:.4f} | "
                    f"LR: {current_lr:.6f}")
        
        scheduler.step()
        
        # Save checkpoint based on validation IoU (weld class)
        if val_epoch_ious[1] > best_val_iou:
            best_val_iou = val_epoch_ious[1]
            torch.save(model.state_dict(), str(SAVE_DIR / "best_model.pth"))
            logging.info(f"Saved best model with Val Weld IoU: {val_epoch_ious[1]:.4f}")
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            
        # Early Stopping check
        early_stopping(val_epoch_ious[1], model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.load_best_model(model)
            break
            
    # Save final model
    torch.save(model.state_dict(), str(SAVE_DIR / "final_model.pth"))
    logging.info(f"Saved final model to {SAVE_DIR / 'final_model.pth'}")
    
    # Calculate final metrics
    final_metrics = calculate_metrics(val_preds_all, val_labels_all)
    final_metrics['best_val_weld_iou'] = best_val_iou
    final_metrics['best_val_accuracy'] = best_acc
    final_metrics['total_epochs'] = len(train_losses)
    
    with open(METRICS_DIR / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Plotting
    epochs_range = range(1, len(train_losses) + 1)
    
    # Combined metrics summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(epochs_range, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs_range, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs_range, train_accuracies, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs_range, val_accuracies, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].plot(epochs_range, train_weld_ious, 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs_range, val_weld_ious, 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('Weld IoU', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Confusion Matrix
    cm = confusion_matrix(val_labels_all, val_preds_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BG', 'Weld'])
    disp.plot(cmap=plt.cm.Blues, ax=axes[1, 1], values_format='d')
    axes[1, 1].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'PointNet++ Training Summary (70:30 Split)\nBest Val Weld IoU: {best_val_iou:.4f} | Best Val Acc: {best_acc:.4f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(METRICS_DIR / 'training_summary.png'), dpi=150)
    plt.close()
    
    # Save epoch metrics to CSV
    import csv
    with open(METRICS_DIR / 'epoch_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_weld_iou', 'train_bg_iou',
                        'val_loss', 'val_acc', 'val_weld_iou', 'val_bg_iou', 'learning_rate'])
        for i in range(len(train_losses)):
            writer.writerow([
                i + 1,
                train_losses[i],
                train_accuracies[i],
                train_weld_ious[i],
                train_ious[i][0],
                val_losses[i],
                val_accuracies[i],
                val_weld_ious[i],
                val_ious[i][0],
                learning_rates[i]
            ])
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY (70:30 SPLIT)")
    print("="*60)
    print(f"Total Epochs: {len(train_losses)}")
    print(f"Best Validation Weld IoU: {best_val_iou:.4f}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Final Weld Precision: {final_metrics['weld_precision']:.4f}")
    print(f"Final Weld Recall: {final_metrics['weld_recall']:.4f}")
    print(f"Final Weld F1: {final_metrics['weld_f1']:.4f}")
    print("="*60)
    print(f"\nMetrics saved to: {METRICS_DIR}")
    print(f"Checkpoints saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
