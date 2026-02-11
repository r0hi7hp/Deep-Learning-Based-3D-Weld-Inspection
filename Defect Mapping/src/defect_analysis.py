"""
Step 8: Defect Analysis Module

Localizes defects and estimates severity based on ISO 5817.
"""
import numpy as np
import logging
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .config import PipelineConfig, SeverityConfig

logger = logging.getLogger(__name__)


@dataclass
class WeldDefect:
    """
    Represents a detected weld defect
    
    Contains localization info and severity estimation
    """
    cluster_id: int
    center_of_mass: List[float]
    num_points: int
    max_deviation: float
    mean_deviation: float
    rmse: float
    severity: str  # "minor", "moderate", "severe", "critical"
    bounding_box_min: List[float]
    bounding_box_max: List[float]
    bounding_box_size: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_cluster_info(cls, info: Dict, severity: str) -> "WeldDefect":
        """Create from cluster info dictionary"""
        return cls(
            cluster_id=info['cluster_id'],
            center_of_mass=info['center_of_mass'],
            num_points=info['num_points'],
            max_deviation=info['max_deviation'],
            mean_deviation=info['mean_deviation'],
            rmse=info['rmse'],
            severity=severity,
            bounding_box_min=info['bounding_box']['min'],
            bounding_box_max=info['bounding_box']['max'],
            bounding_box_size=info['bounding_box']['size']
        )


@dataclass
class DefectReport:
    """
    Complete defect analysis report
    """
    timestamp: str
    reference_file: str
    test_file: str
    model_type: str
    total_points: int
    weld_points: int
    defect_count: int
    defects: List[WeldDefect]
    summary: Dict
    alignment_metrics: Optional[Dict] = None
    filter_statistics: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'reference_file': self.reference_file,
            'test_file': self.test_file,
            'model_type': self.model_type,
            'total_points': self.total_points,
            'weld_points': self.weld_points,
            'defect_count': self.defect_count,
            'defects': [d.to_dict() for d in self.defects],
            'summary': self.summary,
            'alignment_metrics': self.alignment_metrics,
            'filter_statistics': self.filter_statistics
        }
    
    def save_json(self, path: str) -> None:
        """Save report to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved defect report to {path}")
    
    def print_summary(self) -> None:
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("WELD DEFECT ANALYSIS REPORT")
        print("="*60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Model: {self.model_type}")
        print(f"Total Points: {self.total_points:,}")
        print(f"Weld Region Points: {self.weld_points:,}")
        print(f"Defects Found: {self.defect_count}")
        print("-"*60)
        
        if self.defect_count > 0:
            print("\nDefect Details:")
            for i, defect in enumerate(self.defects, 1):
                print(f"\n  [{i}] Defect at ({defect.center_of_mass[0]:.2f}, "
                      f"{defect.center_of_mass[1]:.2f}, {defect.center_of_mass[2]:.2f})")
                print(f"      Severity: {defect.severity.upper()}")
                print(f"      Max Deviation: {defect.max_deviation:.3f} mm")
                print(f"      RMSE: {defect.rmse:.3f} mm")
                print(f"      Points: {defect.num_points}")
        
        print("\n" + "="*60)
        print("Summary by Severity:")
        for severity, count in self.summary['by_severity'].items():
            print(f"  {severity.capitalize()}: {count}")
        print("="*60 + "\n")


class DefectAnalyzer:
    """
    Analyzes detected defect clusters and estimates severity
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the analyzer
        
        Args:
            config: Pipeline configuration with severity thresholds
        """
        self.config = config
        self.severity_config = config.severity
    
    def classify_severity(self, max_deviation: float, rmse: float) -> str:
        """
        Classify defect severity based on ISO 5817 thresholds
        
        Args:
            max_deviation: Maximum point deviation in mm
            rmse: RMSE of cluster deviations
            
        Returns:
            Severity level string
        """
        return self.severity_config.classify_severity(max_deviation, rmse)
    
    def localize_defects(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        distances: np.ndarray,
        cluster_info: List[Dict]
    ) -> List[WeldDefect]:
        """
        Create WeldDefect objects for each cluster
        
        Args:
            points: Point coordinates [N, 3]
            labels: Cluster labels [N]
            distances: Deviation distances [N]
            cluster_info: Pre-computed cluster information
            
        Returns:
            List of WeldDefect objects
        """
        defects = []
        
        for info in cluster_info:
            # Classify severity
            severity = self.classify_severity(
                info['max_deviation'],
                info['rmse']
            )
            
            # Create WeldDefect
            defect = WeldDefect.from_cluster_info(info, severity)
            defects.append(defect)
            
            logger.info(f"Defect {defect.cluster_id}: severity={severity}, "
                       f"max_dev={defect.max_deviation:.3f}, rmse={defect.rmse:.3f}")
        
        return defects
    
    def compute_summary(self, defects: List[WeldDefect]) -> Dict:
        """
        Compute summary statistics for all defects
        
        Args:
            defects: List of detected defects
            
        Returns:
            Summary dictionary
        """
        if not defects:
            return {
                'total_defects': 0,
                'by_severity': {},
                'max_deviation_overall': 0,
                'avg_deviation': 0,
                'total_affected_points': 0,
                'quality_assessment': 'PASS - No defects detected'
            }
        
        # Count by severity
        by_severity = {}
        for defect in defects:
            by_severity[defect.severity] = by_severity.get(defect.severity, 0) + 1
        
        # Compute statistics
        max_deviations = [d.max_deviation for d in defects]
        mean_deviations = [d.mean_deviation for d in defects]
        total_points = sum(d.num_points for d in defects)
        
        # Quality assessment based on ISO 5817 Moderate (Level C)
        has_critical = by_severity.get('critical', 0) > 0
        has_severe = by_severity.get('severe', 0) > 0
        
        if has_critical:
            quality = 'REJECT - Critical defects detected'
        elif has_severe:
            quality = 'REJECT - Severe defects exceed moderate quality limits'
        elif by_severity.get('moderate', 0) > 3:
            quality = 'WARNING - Multiple moderate defects detected'
        else:
            quality = 'PASS - Within ISO 5817 Moderate quality limits'
        
        return {
            'total_defects': len(defects),
            'by_severity': by_severity,
            'max_deviation_overall': float(max(max_deviations)),
            'avg_max_deviation': float(np.mean(max_deviations)),
            'avg_mean_deviation': float(np.mean(mean_deviations)),
            'total_affected_points': int(total_points),
            'quality_assessment': quality
        }
    
    def generate_report(
        self,
        defects: List[WeldDefect],
        reference_path: str,
        test_path: str,
        model_type: str,
        total_points: int,
        weld_points: int,
        alignment_metrics: Optional[Dict] = None,
        filter_statistics: Optional[Dict] = None
    ) -> DefectReport:
        """
        Generate complete defect report
        
        Args:
            defects: List of detected defects
            reference_path: Path to reference model
            test_path: Path to test model
            model_type: DL model type used
            total_points: Total number of points
            weld_points: Number of weld region points
            alignment_metrics: Optional alignment quality metrics
            filter_statistics: Optional filtering statistics
            
        Returns:
            DefectReport object
        """
        summary = self.compute_summary(defects)
        
        report = DefectReport(
            timestamp=datetime.now().isoformat(),
            reference_file=str(reference_path),
            test_file=str(test_path),
            model_type=model_type,
            total_points=total_points,
            weld_points=weld_points,
            defect_count=len(defects),
            defects=defects,
            summary=summary,
            alignment_metrics=alignment_metrics,
            filter_statistics=filter_statistics
        )
        
        return report
    
    def filter_by_severity(
        self,
        defects: List[WeldDefect],
        min_severity: str = "minor"
    ) -> List[WeldDefect]:
        """
        Filter defects by minimum severity
        
        Args:
            defects: List of all defects
            min_severity: Minimum severity to include
            
        Returns:
            Filtered list of defects
        """
        severity_order = ["minor", "moderate", "severe", "critical"]
        min_index = severity_order.index(min_severity)
        
        return [d for d in defects if severity_order.index(d.severity) >= min_index]
