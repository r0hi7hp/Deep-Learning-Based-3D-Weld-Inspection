# Weld Path Detection Web UI

A modern, responsive web interface for detecting weld paths in 3D CAD models using deep learning.

## Features

- **Multiple File Format Support**: Upload `.PLY` or `.STL` CAD models
- **Multi-Architecture Model Support**: Auto-detects and supports various DL architectures:
  - **HybridPointNet** - Standard hybrid PointNet++ and Point Transformer
  - **ImprovedHybridPointNet** - Enhanced 4-level hierarchy with 8-head attention
  - **EnhancedHybridPointNet** - Multi-scale self-attention variant
  - **Point Transformer** - Pure transformer-based architecture
  - **KPConv** - Kernel Point Convolution segmentation
- **Interactive 3D Visualization**: Plotly-based visualization with pan, zoom, and rotate
- **Real-time Statistics**: View point counts, weld percentage, and confidence scores
- **Modern UI**: Glassmorphism design with smooth animations and toast notifications

## Quick Start

### Prerequisites

```bash
pip install flask flask-cors torch numpy open3d plotly
```

### Running the Server

```bash
cd UI
python app.py
```

The server will start at `http://localhost:5000`

### Usage

1. **Upload CAD Model**: Drag-and-drop or click to upload a `.ply` or `.stl` file
2. **Upload DL Model**: Drag-and-drop or click to upload a `.pth` checkpoint file
3. **Detect Weld Path**: Click "Detect Weld Path" button
4. **Explore Results**: Interact with the 3D visualization

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI page |
| `/upload/cad` | POST | Upload CAD model file |
| `/upload/model` | POST | Upload DL model checkpoint |
| `/process` | POST | Run weld detection inference |
| `/status` | GET | Get current upload status |
| `/reset` | POST | Reset all uploaded files |

## Architecture Detection

The system automatically detects the model architecture from checkpoint keys:

| Architecture | Detection Pattern |
|--------------|-------------------|
| Point Transformer | `trans_down`, `trans_up`, `edge_module` |
| KPConv | `kpconv`, `kernel_points` |
| ImprovedHybridPointNet | `pn2_sa1` |
| EnhancedHybridPointNet | `mssa`, `multi_scale` |
| HybridPointNet | `sa1.`, `sa2.` |

## File Structure

```
UI/
├── app.py              # Flask backend with model detection
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── static/
│   ├── css/
│   │   └── style.css   # Modern glassmorphism styles
│   └── js/
│       └── app.js      # Frontend JavaScript
├── templates/
│   └── index.html      # Main HTML template
└── uploads/            # Temporary upload directory
```

## Error Handling

The UI provides robust error handling:

- File size validation (CAD: 100MB max, Model: 500MB max)
- File format validation
- Mesh loading validation with fallbacks
- Model architecture detection with fallbacks
- Toast notifications for all upload/processing events

## Configuration

Edit `app.py` to modify:

```python
MAX_CAD_SIZE_MB = 100    # Maximum CAD file size
MAX_MODEL_SIZE_MB = 500  # Maximum model file size
```

## Troubleshooting

### Model not loading?

1. Ensure the model checkpoint is compatible with one of the supported architectures
2. Check the console logs for detected architecture type
3. Try uploading a different checkpoint from the same architecture

### Visualization empty?

1. Check if your CAD file contains valid geometry
2. Verify the mesh can be loaded by Open3D
3. Check browser console for JavaScript errors

### No weld points detected?

1. The model may not be trained for your specific CAD geometry
2. Try a different model checkpoint
3. Verify the model was trained on similar data
