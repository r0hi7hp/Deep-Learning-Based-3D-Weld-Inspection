"""Weld Processor Package.

A modular, memory-optimized package for processing weld detection datasets.
Aligns paired 3D meshes, computes weld candidate points, and saves labeled point clouds.
"""
from config import ProcessingConfig
from processor import WeldProcessor

__all__ = ["ProcessingConfig", "WeldProcessor"]
__version__ = "1.0.0"
