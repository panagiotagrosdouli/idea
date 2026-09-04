from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_names(split: str, start: int, stop: int, total: int) -> list[str]:
    return [f"{split}.tfrecord-{i:05d}-of-{total:05d}" for i in range(start, stop)]


def inspect_split(root: Path, spec: dict) -> dict:
    names = expected_names(
        spec["split"],
        int(spec["selected_shard_start"]),
        int(spec["selected_shard_stop_exclusive"]),
        int(spec["total_shards"]),
    )
    records = []
    missing = []
    for name in names:
        path = root / spec["split"] / name
        if not path.is_file():
            missing.append(name)
            continue
        records.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "split": spec["split"],
        "expected_count": len(names),
        "present_count": len(records),
        "complete": not missing,
        "missing": missing,
        "shards": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize immutable provenance for the frozen WOMD shard selection.")
    parser.add_argument("data_root", help="Directory containing training/ and validation/ TFRecord directories")
    parser.add_argument("--config", default="configs/womd_paper_corpus.json")
    parser.add_argument("--output", default="artifacts/paper_final/01_data/source_shard_manifest.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(args.data_root)
    training = inspect_split(root, config["training"])
    validation = inspect_split(root, config["validation"])
    report = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "version": config["version"],
        "source": config["source"],
        "config_sha256": sha256(config_path),
        "training": training,
        "validation": validation,
        "complete": training["complete"] and validation["complete"],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
