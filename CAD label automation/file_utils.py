"""File and directory utility functions."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class FileUtils:
    """Utility class for file and directory operations."""

    @staticmethod
    def find_mesh_file(model_dir: Path, token: str) -> Path:
        """Find a mesh file containing the given token in its name.
        
        Args:
            model_dir: Directory to search.
            token: Substring to match in filename.
            
        Returns:
            Path to the matching mesh file.
            
        Raises:
            FileNotFoundError: If no matching file is found.
        """
        token = token.lower().replace(" ", "")
        for ext in ("*.ply", "*.stl"):
            for candidate in model_dir.glob(ext):
                stem = candidate.stem.lower().replace(" ", "")
                if token in stem:
                    return candidate
        raise FileNotFoundError(
            f"No PLY/STL file containing '{token}' in {model_dir}"
        )

    @staticmethod
    def ensure_destination(base_output: Optional[Path], model_dir: Path) -> Path:
        """Ensure output directory exists and return the path.
        
        Args:
            base_output: Base output directory (None to use model_dir).
            model_dir: Model directory being processed.
            
        Returns:
            Path to the output directory.
        """
        if base_output is None:
            return model_dir
        destination = base_output / model_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    @staticmethod
    def iterate_model_directories(base_dir: Path) -> Iterator[Path]:
        """Iterate over model subdirectories.
        
        Args:
            base_dir: Parent directory containing model folders.
            
        Yields:
            Paths to model subdirectories.
        """
        for entry in sorted(base_dir.iterdir()):
            if entry.is_dir():
                yield entry
