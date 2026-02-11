"""Command-line interface for weld processing.

Usage:
    python -m weld_processor.main <base_dir> [options]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import ProcessingConfig
from processor import WeldProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process weld model folders and generate labeled NPZ files.",
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing model subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to model folder.",
    )
    parser.add_argument(
        "--reference-token",
        default="part1",
        help="Substring identifying reference mesh.",
    )
    parser.add_argument(
        "--welded-token",
        default="part2",
        help="Substring identifying welded mesh.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if NPZ exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--icp-samples",
        type=int,
        default=ProcessingConfig.icp_samples,
        help="Samples for ICP alignment.",
    )
    parser.add_argument(
        "--icp-max-distance",
        type=float,
        default=ProcessingConfig.icp_max_distance,
        help="Max correspondence distance for ICP.",
    )
    parser.add_argument(
        "--weld-samples",
        type=int,
        default=ProcessingConfig.weld_sample_count,
        help="Points sampled for weld detection.",
    )
    parser.add_argument(
        "--weld-threshold",
        type=float,
        default=ProcessingConfig.weld_distance_threshold,
        help="Distance threshold for weld candidates.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=ProcessingConfig.dbscan_eps,
        help="DBSCAN epsilon parameter.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=ProcessingConfig.dbscan_min_samples,
        help="DBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--full-samples",
        type=int,
        default=ProcessingConfig.full_sample_count,
        help="Points for final labeled cloud.",
    )
    parser.add_argument(
        "--weld-radius",
        type=float,
        default=ProcessingConfig.weld_label_radius,
        help="Radius for labeling weld points.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s - %(message)s",
    )

    config = ProcessingConfig(
        icp_samples=args.icp_samples,
        icp_max_distance=args.icp_max_distance,
        weld_sample_count=args.weld_samples,
        weld_distance_threshold=args.weld_threshold,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        full_sample_count=args.full_samples,
        weld_label_radius=args.weld_radius,
    )

    processor = WeldProcessor(config)
    processor.process_batch(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        reference_token=args.reference_token,
        welded_token=args.welded_token,
    )


if __name__ == "__main__":
    main()
