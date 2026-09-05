"""Dataset-free synthetic vehicular simulation protocol."""

from .mobility import Scenario, SyntheticMobilityConfig, generate_scenario

__all__ = ["Scenario", "SyntheticMobilityConfig", "generate_scenario"]
