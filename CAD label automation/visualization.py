"""Visualization utilities for weld detection results."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Lazy import matplotlib
_plt = None


def _get_pyplot():
    """Lazy load matplotlib.pyplot."""
    global _plt
    if _plt is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _plt = plt
        except ImportError:
            pass
    return _plt


class Visualizer:
    """Handles visualization of point clouds and weld regions."""

    @staticmethod
    def render_visualization(
        base_points: NDArray[np.floating],
        weld_points: NDArray[np.floating],
        destination: Path,
    ) -> Optional[Path]:
        """Render and save a 3D visualization of the point cloud.
        
        Args:
            base_points: Full point cloud array.
            weld_points: Weld region points to highlight.
            destination: Output file path.
            
        Returns:
            Path to saved image, or None if matplotlib unavailable.
        """
        plt = _get_pyplot()
        if plt is None:
            logger.warning("Matplotlib not installed; skipping visualization")
            return None

        base_points = np.asarray(base_points)
        if base_points.size == 0:
            logger.warning("No points for visualization")
            return None

        destination.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot base points
        ax.scatter(
            base_points[:, 0],
            base_points[:, 1],
            base_points[:, 2],
            c="lightgray",
            s=1,
            alpha=0.3,
        )

        # Plot weld points
        if len(weld_points) > 0:
            weld_points = np.asarray(weld_points)
            ax.scatter(
                weld_points[:, 0],
                weld_points[:, 1],
                weld_points[:, 2],
                c="red",
                s=6,
                alpha=0.9,
            )

        ax.set_title("Weld Region Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set axis limits
        center = base_points.mean(axis=0)
        max_range = (base_points.max(axis=0) - base_points.min(axis=0)).max()
        if not np.isfinite(max_range) or max_range == 0:
            max_range = 1.0
        radius = max_range * 0.6
        for axis_setter, coord in zip(
            (ax.set_xlim, ax.set_ylim, ax.set_zlim), center
        ):
            axis_setter(coord - radius, coord + radius)

        ax.view_init(elev=20, azim=45)
        fig.tight_layout()
        fig.savefig(destination, dpi=300)
        plt.close(fig)

        logger.info("Saved visualization: %s", destination)
        return destination
