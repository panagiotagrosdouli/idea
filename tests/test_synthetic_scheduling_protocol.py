from predictive_pc_fmcw.synthetic.scheduling_protocol import (
    SCHEDULER_FAMILIES,
    scheduler_protocol_manifest,
    validate_scheduler_protocol,
)


def test_scheduler_protocol_is_exact_s0_through_s4() -> None:
    validate_scheduler_protocol()
    assert [item.protocol_id for item in SCHEDULER_FAMILIES] == [
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
    ]


def test_scheduler_protocol_preserves_scientific_roles() -> None:
    by_id = {item.protocol_id: item for item in SCHEDULER_FAMILIES}
    assert by_id["S0"].scheduler_name == "reactive_greedy"
    assert by_id["S1"].scheduler_name == "cv_predictive"
    assert by_id["S2"].learned_objective == "trajectory_only"
    assert by_id["S3"].learned_objective == "full_communication_aware"
    assert by_id["S4"].scheduler_name == "oracle"
    assert by_id["S4"].deployable is False


def test_scheduler_protocol_declares_paired_scenario_unit() -> None:
    manifest = scheduler_protocol_manifest()
    assert manifest["inferential_unit"] == "scenario_episode"
    assert set(manifest["paired_inputs"]) == {
        "mobility_scenario",
        "channel_realization",
        "packet_arrivals",
        "packet_deadlines",
        "traffic_seed",
    }
