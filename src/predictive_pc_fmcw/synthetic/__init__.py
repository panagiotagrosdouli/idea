"""Dataset-free synthetic vehicular simulation protocol."""

from .baselines import (
    BaselineMetrics,
    evaluate_synthetic_baselines,
    save_baseline_results,
)
from .checkpoint_selection import (
    build_development_checkpoint_selection,
    select_representative_seed,
)
from .dataset import DatasetBuildConfig, build_dataset, validate_dataset
from .episodes import SyntheticEpisode, compose_synthetic_episodes
from .freeze import verify_publication_training_freeze
from .link_evaluation import (
    LinkPredictionMetrics,
    evaluate_synthetic_link_prediction,
    save_link_prediction_results,
)
from .mobility import Scenario, SyntheticMobilityConfig, generate_scenario
from .observations import CausalObservations, ObservationNoiseConfig, observe_scenario
from .official_export import build_official_evaluation_npz
from .operating_region import (
    OPERATING_CONDITIONS,
    OperatingCondition,
    analyze_operating_region,
    operating_region_protocol_manifest,
    run_operating_region_sweep,
    validate_operating_region_protocol,
)
from .robustness import (
    ROBUSTNESS_CONDITIONS,
    RobustnessCondition,
    robustness_protocol_manifest,
    run_synthetic_robustness_sweep,
    validate_robustness_protocol,
)
from .scheduling_evaluation import (
    build_scheduling_plan,
    run_synthetic_scheduling_evaluation,
)
from .scheduling_protocol import (
    PAIRED_TRAFFIC_SEEDS,
    SCHEDULER_FAMILIES,
    SchedulerFamily,
    scheduler_protocol_manifest,
    validate_scheduler_protocol,
)
from .scheduling_statistics import (
    PRIMARY_COMPARISONS,
    SCHEDULING_METRICS,
    aggregate_traffic_seeds,
    analyze_scheduling_file,
    analyze_scheduling_rows,
)
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
    "OPERATING_CONDITIONS",
    "ObservationNoiseConfig",
    "OperatingCondition",
    "PAIRED_TRAFFIC_SEEDS",
    "PRIMARY_COMPARISONS",
    "ROBUSTNESS_CONDITIONS",
    "SCHEDULER_FAMILIES",
    "SCHEDULING_METRICS",
    "RobustnessCondition",
    "Scenario",
    "SchedulerFamily",
    "SplitManifest",
    "SyntheticEpisode",
    "SyntheticMobilityConfig",
    "aggregate_traffic_seeds",
    "analyze_operating_region",
    "analyze_scheduling_file",
    "analyze_scheduling_rows",
    "build_dataset",
    "build_development_checkpoint_selection",
    "build_official_evaluation_npz",
    "build_scheduling_plan",
    "build_split_manifest",
    "build_synthetic_training_npz",
    "compose_synthetic_episodes",
    "evaluate_synthetic_baselines",
    "evaluate_synthetic_link_prediction",
    "generate_scenario",
    "observe_scenario",
    "operating_region_protocol_manifest",
    "robustness_protocol_manifest",
    "run_operating_region_sweep",
    "run_synthetic_robustness_sweep",
    "run_synthetic_scheduling_evaluation",
    "save_baseline_results",
    "save_link_prediction_results",
    "scheduler_protocol_manifest",
    "select_representative_seed",
    "validate_dataset",
    "validate_operating_region_protocol",
    "validate_robustness_protocol",
    "validate_scheduler_protocol",
    "validate_split_manifest",
    "validate_synthetic_training_npz",
    "verify_publication_training_freeze",
]
