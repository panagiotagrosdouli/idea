"""Dataset-free synthetic vehicular simulation protocol."""

from .dataset import DatasetBuildConfig, build_dataset, validate_dataset
from .mobility import Scenario, SyntheticMobilityConfig, generate_scenario
from .observations import CausalObservations, ObservationNoiseConfig, observe_scenario
from .splits import SplitManifest, build_split_manifest, validate_split_manifest
from .training_export import (
    build_synthetic_training_npz,
    validate_synthetic_training_npz,
)

__all__ = [
    "CausalObservations",
    "DatasetBuildConfig",
    "ObservationNoiseConfig",
    "Scenario",
    "SplitManifest",
    "SyntheticMobilityConfig",
    "build_dataset",
    "build_split_manifest",
    "build_synthetic_training_npz",
    "generate_scenario",
    "observe_scenario",
    "validate_dataset",
    "validate_split_manifest",
    "validate_synthetic_training_npz",
]
