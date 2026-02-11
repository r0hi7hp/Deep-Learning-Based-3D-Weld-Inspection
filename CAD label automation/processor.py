"""Main weld processor class orchestrating the pipeline."""
from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from config import ProcessingConfig
from file_utils import FileUtils
from mesh_operations import MeshOperations
from visualization import Visualizer
from weld_detection import WeldDetector

logger = logging.getLogger(__name__)


class WeldProcessor:
    """Orchestrates the weld detection and labeling pipeline."""

    def __init__(self, config: ProcessingConfig) -> None:
        """Initialize the processor with configuration.
        
        Args:
            config: Processing configuration parameters.
        """
        self._config = config

    @property
    def config(self) -> ProcessingConfig:
        """Get the processing configuration."""
        return self._config

    def process_model_directory(
        self,
        model_dir: Path,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
        reference_token: str = "part1",
        welded_token: str = "part2",
        label_stem: str = "label",
    ) -> Optional[Path]:
        """Process a single model directory.
        
        Args:
            model_dir: Directory containing mesh files.
            output_dir: Output directory (None to use model_dir).
            overwrite: Whether to overwrite existing outputs.
            reference_token: Token to identify reference mesh.
            welded_token: Token to identify welded mesh.
            label_stem: Stem for output filenames.
            
        Returns:
            Path to generated NPZ file, or None if skipped/failed.
        """
        try:
            reference_path = FileUtils.find_mesh_file(model_dir, reference_token)
            welded_path = FileUtils.find_mesh_file(model_dir, welded_token)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            return None

        output_root = FileUtils.ensure_destination(output_dir, model_dir)
        npz_path = output_root / f"{label_stem}.npz"

        if npz_path.exists() and not overwrite:
            logger.info("Skipping %s (already processed)", model_dir.name)
            return npz_path

        # Align meshes
        aligned_mesh, reference_mesh = MeshOperations.align_meshes(
            welded_path,
            reference_path,
            samples=self._config.icp_samples,
            max_distance=self._config.icp_max_distance,
        )

        # Detect weld points
        weld_points = WeldDetector.compute_weld_points(
            aligned_mesh,
            reference_mesh,
            sample_count=self._config.weld_sample_count,
            distance_threshold=self._config.weld_distance_threshold,
            dbscan_eps=self._config.dbscan_eps,
            dbscan_min_samples=self._config.dbscan_min_samples,
        )

        # Sample full point cloud
        full_points = MeshOperations.sample_points(
            aligned_mesh,
            self._config.full_sample_count,
        )

        # Cleanup meshes after sampling
        del aligned_mesh, reference_mesh
        gc.collect()

        # Label points
        labels = WeldDetector.label_points(
            full_points,
            weld_points,
            weld_radius=self._config.weld_label_radius,
        )

        # Save output
        output_root.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, points=full_points, labels=labels)
        logger.info("Saved: %s", npz_path)

        # Generate visualization
        Visualizer.render_visualization(
            base_points=full_points,
            weld_points=weld_points,
            destination=output_root / f"{label_stem}_visualization.png",
        )

        # Cleanup
        del full_points, weld_points, labels
        gc.collect()

        return npz_path

    def process_batch(
        self,
        base_dir: Path,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
        reference_token: str = "part1",
        welded_token: str = "part2",
    ) -> list[Path]:
        """Process all model directories in a base directory.
        
        Args:
            base_dir: Directory containing model subdirectories.
            output_dir: Output directory (None to use model directories).
            overwrite: Whether to overwrite existing outputs.
            reference_token: Token to identify reference meshes.
            welded_token: Token to identify welded meshes.
            
        Returns:
            List of generated NPZ file paths.
        """
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")

        processed_files: list[Path] = []

        for model_dir in FileUtils.iterate_model_directories(base_dir):
            logger.info("Processing: %s", model_dir.name)

            # Determine label stem
            match = re.search(r"\d+", model_dir.name)
            stem_suffix = match.group() if match else model_dir.name

            result = self.process_model_directory(
                model_dir,
                output_dir=output_dir,
                overwrite=overwrite,
                reference_token=reference_token,
                welded_token=welded_token,
                label_stem=f"label_{stem_suffix}",
            )

            if result is not None:
                processed_files.append(result)

            # Force garbage collection between models
            gc.collect()

        logger.info("Processed %d models", len(processed_files))
        return processed_files
