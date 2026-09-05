"""Dataset-free synthetic vehicular simulation protocol."""

from .baselines import (
    BaselineMetrics,
    evaluate_synthetic_baselines,
    save_baseline_results,
)
from .dataset import DatasetBuildConfig, build_dataset, validate_dataset
from .link_evaluation import (
    LinkPredictionMetrics,
    evaluate_synthetic_link_prediction,
    save_link_prediction_results,
)
from .mobility import Scenario, SyntheticMobilityConfig, generate_scenario
from .observations import CausalObservations, ObservationNoiseConfig, observe_scenario
from .splits import SplitManifest, build_split_manifest, validate_split_manifest
from .training_export import (
    build_synthetic_training_npz,
    validate_synthetic_training_npz,
)

__all__ = [
    "BaselineMetrics",
    "CausalObservations",
    "DatasetBuildConfig",
    "LinkPredictionMetrics",
    "ObservationNoiseConfig",
    "Scenario",
    "SplitManifest",
    "SyntheticMobilityConfig",
    "build_dataset",
    "build_split_manifest",
    "build_synthetic_training_npz",
    "evaluate_synthetic_baselines",
    "evaluate_synthetic_link_prediction",
    "generate_scenario",
    "observe_scenario",
    "save_baseline_results",
    "save_link_prediction_results",
    "validate_dataset",
    "validate_split_manifest",
    "validate_synthetic_training_npz",
]
