"""
Comparative Experiments Framework

Provides the structure for comparing tracker configurations,
detector variants, and confidence thresholds. Currently NOT MEASURED
because no ground-truth dataset is available.
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")


@dataclass
class ExperimentConfig:
    """A single experiment configuration to compare."""
    name: str
    description: str
    tracker_type: str = "yolo_kalman"       # "yolo_kalman" | "yolo_flow_kalman"
    detector_weights: str = "yolo11n.pt"
    conf_threshold: float = 0.1
    seed_min_conf: float = 0.3
    optical_flow: bool = False
    confidence_gate: bool = True


@dataclass
class ExperimentResult:
    """Results from running one configuration on the evaluation dataset."""
    config: ExperimentConfig
    detection: Dict = field(default_factory=dict)
    tracking: Dict = field(default_factory=dict)
    release_frame: Dict = field(default_factory=dict)
    status: str = "NOT_MEASURED"


# Default experiments to compare
DEFAULT_EXPERIMENTS = [
    ExperimentConfig(
        name="Baseline",
        description="YOLO + Kalman, production settings",
        tracker_type="yolo_kalman",
        conf_threshold=0.1,
        optical_flow=False,
        confidence_gate=True,
    ),
    ExperimentConfig(
        name="YOLO + Flow + Kalman",
        description="YOLO + optical flow + Kalman (experimental)",
        tracker_type="yolo_flow_kalman",
        conf_threshold=0.1,
        optical_flow=True,
        confidence_gate=True,
    ),
    ExperimentConfig(
        name="High confidence gate",
        description="YOLO + Kalman with strict confidence gate (0.25)",
        tracker_type="yolo_kalman",
        conf_threshold=0.1,
        optical_flow=False,
        confidence_gate=True,
    ),
    ExperimentConfig(
        name="No confidence gate",
        description="YOLO + Kalman without confidence gating",
        tracker_type="yolo_kalman",
        conf_threshold=0.1,
        optical_flow=False,
        confidence_gate=False,
    ),
]


def run_comparative_experiment(configs: List[ExperimentConfig] = None) -> dict:
    """
    Run comparative experiments on the evaluation dataset.

    Currently returns NOT_MEASURED for all configurations because
    no ground-truth data is available.
    """
    if configs is None:
        configs = DEFAULT_EXPERIMENTS

    results = {}
    for cfg in configs:
        results[cfg.name] = ExperimentResult(
            config=cfg,
            status="NOT_MEASURED",
            detection={"status": "NOT_MEASURED"},
            tracking={"status": "NOT_MEASURED"},
            release_frame={"status": "NOT_MEASURED"},
        )

    return {
        "status": "NOT_MEASURED",
        "message": ("Comparative experiments require ground-truth annotated data "
                    "in evaluation/videos/ and evaluation/annotations/."),
        "configurations_tested": len(configs),
        "results": {name: {"status": r.status} for name, r in results.items()},
    }


def format_comparison_table(results: dict) -> str:
    """Format results as a markdown table."""
    if results.get("status") == "NOT_MEASURED":
        return ("| Configuration | mAP50 | Recall | ID Switches | Release Error |\n"
                "|---|---|---|---|---|\n"
                "| Baseline | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |\n"
                "| YOLO + Flow + Kalman | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |\n"
                "| Cricket YOLO | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |")

    lines = ["| Configuration | mAP50 | Recall | ID Switches | Release Error |",
             "|---|---|---|---|---|"]
    for name, data in results.get("results", {}).items():
        det = data.get("detection", {})
        trk = data.get("tracking", {})
        rel = data.get("release_frame", {})
        lines.append(
            f"| {name} | "
            f"{det.get('ap50', 'N/A')} | "
            f"{det.get('recall', 'N/A')} | "
            f"{trk.get('id_switches', 'N/A')} | "
            f"{rel.get('absolute_error', 'N/A')} |"
        )
    return "\n".join(lines)
