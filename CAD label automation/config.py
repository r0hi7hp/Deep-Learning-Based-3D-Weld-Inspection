"""Configuration dataclass for weld processing parameters."""
from __future__ import annotations

from dataclasses import dataclass    


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for weld processing pipeline.
    
    Attributes:
        icp_samples: Number of Poisson-disk samples for ICP alignment.
        icp_max_distance: Maximum correspondence distance for ICP.
        weld_sample_count: Points sampled to detect weld differences.
        weld_distance_threshold: Distance threshold to classify weld candidates.
        dbscan_eps: DBSCAN epsilon parameter for clustering.
        dbscan_min_samples: DBSCAN minimum samples parameter.
        full_sample_count: Points sampled for final labeled cloud.
        weld_label_radius: Radius for labeling weld points.
    """
    icp_samples: int = 15_000
    icp_max_distance: float = 2.0
    weld_sample_count: int = 30_000
    weld_distance_threshold: float = 1.0
    dbscan_eps: float = 5.0
    dbscan_min_samples: int = 10
    full_sample_count: int = 20_000
    weld_label_radius: float = 1.0
