"""
Configuration file for Point Transformer model (WITH DATA AUGMENTATION)
Centralized hyperparameters for training and inference
"""
import os


class Config:
    """Configuration class for Point Transformer model"""
    
    # ========== Paths ==========
    # Dataset
    DATASET_ROOT = "C:/Users/rohith.p/Desktop/Weld Inspection/Weld Path/Dataset"
    
    # Model checkpoints and outputs (relative to script directory)
    CHECKPOINT_DIR = "checkpoints"
    METRICS_DIR = "Metrics"
    
    # ========== Model Architecture ==========
    NUM_CLASSES = 2  # Background and weld
    NUM_HEADS = 8    # Multi-head attention
    NUM_POINTS = 2048  # Number of points to sample from each model
    
    # ========== Training Parameters ==========
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler
    LR_SCHEDULER = 'cosine'  # 'cosine' or 'step'
    LR_DECAY_STEP = 20
    LR_DECAY_RATE = 0.7
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Gradient accumulation
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
    
    # ========== Loss Weights ==========
    FOCAL_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    AUX_WEIGHT = 0.3
    BOUNDARY_WEIGHT = 0.1
    
    # ========== Data Augmentation ==========
    USE_AUGMENTATION = True  # ENABLED for this version
    RANDOM_ROTATION = True
    RANDOM_SCALE = True
    RANDOM_JITTER = True
    RANDOM_DROPOUT = True
    RANDOM_MIRROR = True
    
    # Augmentation parameters
    SCALE_LOW = 0.8
    SCALE_HIGH = 1.25
    JITTER_SIGMA = 0.01
    JITTER_CLIP = 0.05
    DROPOUT_RATIO = 0.1
    SHIFT_RANGE = 0.1
    
    # ========== Dataset Parameters ==========
    TRAIN_SPLIT = 0.7  # 70% training, 30% validation
    BALANCE_CLASSES = False
    TARGET_WELD_RATIO = 0.4
    
    # ========== Device ==========
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    NUM_WORKERS = 0
    PIN_MEMORY = True
    
    # ========== Logging ==========
    PRINT_FREQ = 10  # Print every N batches
    SAVE_FREQ = 5    # Save checkpoint every N epochs
    
    # ========== Prediction ==========
    PRED_BATCH_SIZE = 1
    SAVE_NPZ = True
    SAVE_VISUALIZATION = False
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 60)
        print("Configuration Parameters (WITH DATA AUGMENTATION)")
        print("=" * 60)
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(cls, attr)
                print(f"{attr:30s}: {value}")
        print("=" * 60)
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.METRICS_DIR, exist_ok=True)
        print(f"Created directories:")
        print(f"  - {cls.CHECKPOINT_DIR}")
        print(f"  - {cls.METRICS_DIR}")


if __name__ == '__main__':
    # Test configuration
    Config.print_config()
    Config.create_dirs()
