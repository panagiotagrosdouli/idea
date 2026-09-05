import hashlib
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.synthetic.configuration import (
    load_synthetic_protocol_config,
)


def test_canonical_synthetic_protocol_loads_without_default_drift() -> None:
    path = Path("configs/synthetic_dataset_v1.json")
    loaded = load_synthetic_protocol_config(path)
    config = loaded.build_config
    assert config.master_seed == 20260905
    assert config.mobility.duration_s == 12.0
    assert config.mobility.sampling_hz == 10.0
    assert config.mobility.speed_mps == (5.0, 35.0)
    assert config.ood_mobility.speed_mps == (35.0, 45.0)
    assert config.observations.range_std_m == 0.05
    assert np.isclose(config.observations.bearing_std_rad, np.deg2rad(0.10))
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
