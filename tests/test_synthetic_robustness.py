from predictive_pc_fmcw.config import LinkConfig, SensingConfig
from predictive_pc_fmcw.synthetic.robustness import (
    ROBUSTNESS_CONDITIONS,
    _condition_forecast_link,
    _condition_sensing,
    robustness_protocol_manifest,
    validate_robustness_protocol,
)


def test_robustness_protocol_is_frozen_and_unique() -> None:
    validate_robustness_protocol()
    names = [condition.name for condition in ROBUSTNESS_CONDITIONS]
    assert names[0] == "nominal"
    assert len(names) == len(set(names))
    manifest = robustness_protocol_manifest()
    assert manifest["actual_channel_frozen_across_conditions"] is True


def test_noise_condition_changes_only_declared_sensing_uncertainty() -> None:
    nominal = SensingConfig(
        model="range_bearing_assumed",
        range_std_base_m=0.05,
        range_std_per_m=0.0,
        bearing_std_deg=0.1,
    )
    condition = next(
        item for item in ROBUSTNESS_CONDITIONS if item.name == "observation_noise_x4"
    )
    changed = _condition_sensing(nominal, condition)
    assert changed.range_std_base_m == 0.2
    assert changed.bearing_std_deg == 0.4
    assert changed.model == nominal.model


def test_forecast_channel_mismatch_does_not_mutate_actual_config() -> None:
    actual = LinkConfig()
    condition = next(
        item for item in ROBUSTNESS_CONDITIONS if item.name == "forecast_range_only"
    )
    forecast = _condition_forecast_link(actual, condition)
    assert actual.channel_mode == "full"
    assert forecast.channel_mode == "range_only"
    assert actual.reference_snr_db == forecast.reference_snr_db
