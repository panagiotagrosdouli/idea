.PHONY: install test validate benchmark womd ablation matrix

install:
	python -m pip install -e ".[dev]"

test:
	python -m unittest discover -s tests -v

validate:
	pcfmcw validate --config configs/default.json --output artifacts/validation.json

benchmark:
	pcfmcw benchmark --config configs/default.json --output artifacts/synthetic_benchmark

womd:
	pcfmcw benchmark --config configs/default.json \
		--womd-export data/example/womd_trajectories.json \
		--max-vehicles 5 --output artifacts/womd_proxy_benchmark

ablation:
	pcfmcw ablation --config configs/default.json \
		--output artifacts/horizon_ablation --horizons 3 5 10 20

matrix:
	pcfmcw matrix --config configs/default.json \
		--matrix configs/experiment_matrix.json \
		--output artifacts/experiment_matrix

