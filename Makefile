.PHONY: install test lint validate benchmark womd ablation matrix paper-quick \
	paper-full motion manifest paper-ablation staged corrected-quick \
	corrected-full paper-draft reproducibility reproduce

install:
	python -m pip install -e ".[dev]"

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

lint:
	ruff check src tests scripts

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

motion:
	pcfmcw motion-eval --config configs/default.json \
		--output artifacts/motion_baselines

manifest:
	python scripts/00_freeze_paper_manifest.py

paper-ablation:
	pcfmcw paper-ablation --config configs/default.json \
		--ber-lut artifacts/ber/dbpsk_ber_lut.csv \
		--output artifacts/paper_run/paper_ablations

paper-quick:
	PYTHONPATH=src python scripts/run_paper_pipeline.py --quick \
		--output artifacts/paper_run
	PYTHONPATH=src python scripts/build_paper_pdf.py

paper-full:
	PYTHONPATH=src python scripts/run_paper_pipeline.py \
		--output artifacts/paper_run_full

staged:
	PYTHONPATH=src python scripts/07_run_staged_experiments.py \
		--config configs/default.json \
		--output artifacts/staged_experiments

corrected-quick:
	PYTHONPATH=src python scripts/run_corrected_pipeline.py --quick \
		--output artifacts/corrected_v2_quick

corrected-full:
	PYTHONPATH=src python scripts/run_corrected_pipeline.py \
		--output artifacts/corrected_v2

paper-draft:
	PYTHONPATH=src python scripts/build_paper_pdf.py

reproducibility:
	PYTHONPATH=src python scripts/build_reproducibility_manifest.py \
		--output artifacts/corrected_v2/reproducibility_manifest.json

reproduce: test lint corrected-quick paper-draft reproducibility
