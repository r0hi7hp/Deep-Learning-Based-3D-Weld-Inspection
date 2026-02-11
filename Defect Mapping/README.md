# Weld Defect Mapping Pipeline

A comprehensive pipeline for mapping weld defects to segmented weld regions using deep learning and geometric analysis.

## Features

- **Multi-model Support**: PointNet++, KPConv, and other DL architectures
- **8-Step Pipeline**: From weld segmentation to defect severity estimation
- **ISO 5817 Compliance**: Moderate quality thresholds for defect classification
- **Interactive Visualization**: 3D defect visualization with Plotly

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run defect mapping
python main.py --reference Models/model_1/actual.ply \
               --test Models/model_1/scan.ply \
               --model-type pointnet++ \
               --output results/

# Or run with KPConv
python main.py --reference Models/model_1/actual.ply \
               --test Models/model_1/scan.ply \
               --model-type kpconv \
               --output results/
```

## Project Structure

```
Defect Mapping/
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
├── src/
│   ├── config.py              # Configuration
│   ├── weld_segmentation.py   # Step 1: DL Weld Segmentation
│   ├── point_cloud_prep.py    # Step 2: Point Cloud Preparation
│   ├── alignment.py           # Step 3: FPFH + ICP Alignment
│   ├── distance_computation.py # Step 4: NN Distance
│   ├── defect_filter.py       # Steps 5-6: Weld-constrained filtering
│   ├── clustering.py          # Step 7: DBSCAN Clustering
│   ├── defect_analysis.py     # Step 8: Defect Analysis
│   ├── pipeline.py            # Main orchestration
│   └── visualization.py       # 3D Visualization
├── Models/                    # Sample data
│   └── model_1/
│       ├── actual.ply         # Reference (defect-free)
│       └── scan.ply           # Test (with defects)
└── results/                   # Output directory
```

## Pipeline Steps

1. **Weld Segmentation**: Uses trained DL model (PointNet++/KPConv) to identify weld regions
2. **Point Cloud Preparation**: Uniform sampling for both reference and test clouds
3. **Alignment**: FPFH coarse + ICP fine alignment
4. **Distance Computation**: Nearest-neighbor distance calculation
5. **Weld Filtering**: Apply weld mask to focus on weld region deviations
6. **Deviation Selection**: Select top percentile of deviations
7. **DBSCAN Clustering**: Cluster high-deviation points into defect regions
8. **Defect Analysis**: Localize defects and estimate severity per ISO 5817
