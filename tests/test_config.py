import json
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.config import config_from_dict, load_config


class ConfigTest(unittest.TestCase):
    def test_nested_config_and_tuple_conversion(self):
        config = config_from_dict(
            {"benchmark": {"episodes": 2, "schedulers": ["random", "oracle"]}}
        )
        self.assertEqual(config.benchmark.episodes, 2)
        self.assertEqual(config.benchmark.schedulers, ("random", "oracle"))

    def test_round_trip_file(self):
        config = config_from_dict({})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(loaded, config)


if __name__ == "__main__":
    unittest.main()

