# Weld Detection Pipeline

A modular, memory-optimized Python tool for detecting and labeling weld regions by comparing 3D CAD models.

## How it Works
1. **Aligns** two 3D meshes (a reference part and a welded part) using ICP (Iterative Closest Point).
2. **Compares** the surfaces to find points that deviate (potential welds).
3. **Filters** noise using DBSCAN clustering.
4. **Saves** the results as labeled point clouds (`.npz`) and 3D visualization images (`.png`).

---

## Quick Start
```powershell
# 1. Setup environment
conda env create -f environment.yml
conda activate weld-inspection

# 2. Run the tool
python main.py "D:\path\to\models_directory"
```

---

## Input & Output
**Input Structure:**
The tool expects a base directory containing subfolders. Each subfolder should have two mesh files (PLY or STL):
- One file identifying the reference (default: contains "part1")
- One file identifying the welded part (default: contains "part2")

**Output Files:**
For each processed model, it generates:
- **`label_*.npz`**: Compressed numpy file with:
  - `points`: (N, 3) float array of XYZ coordinates.
  - `labels`: (N,) uint8 array (0 = background, 1 = weld).
- **`*_visualization.png`**: 3D scatter plot showing the detected weld in red.

---

## Command Line Options
Complete usage: `python main.py <base_dir> [options]`

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | Same as input | Destination folder for results |
| `--overwrite` | `False` | Force re-processing if output exists |
| `--weld-threshold` | `1.0` | Distance threshold to consider a point a weld |
| `--icp-samples` | `15000` | Number of points used for alignment accuracy |
| `--full-samples` | `20000` | Number of points in the final output cloud |
| `--log-level` | `INFO` | Set to `DEBUG` for verbose output |

---

## Project Structure
This tool is built as a Python package with the following components:

- **`main.py`**: CLI entry point.
- **`processor.py`**: Orchestrates the pipeline (finding files -> aligning -> detecting -> saving).
- **`config.py`**: Central place for all algorithm parameters (sample counts, thresholds).
- **`mesh_operations.py`**: Handles 3D mesh loading and mathematical alignment.
- **`weld_detection.py`**: Core logic for finding weld points using KDTree and clustering.
- **`visualization.py`**: Generates the 3D preview images.
- **`file_utils.py`**: Helper functions for directory crawling.

---

## Python API
You can also import the package in your own scripts:

```python
from pathlib import Path
from config import ProcessingConfig
from processor import WeldProcessor

# Custom settings
config = ProcessingConfig(weld_distance_threshold=0.8)
processor = WeldProcessor(config)

# Process a single folder
processor.process_model_directory(
    Path("D:/models/sample_01"),
    reference_token="part1",
    welded_token="part2"
)
```
