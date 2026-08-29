from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import LinkConfig
from ..data.manifest import sha256_file


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: str
    best_epoch: int
    validation_loss: float
    validation_ade_m: float
    validation_fde_m: float
    validation_trajectory_loss: float
    validation_link_loss: float
    validation_outage_loss: float
    train_samples: int
    validation_samples: int
    objective: str
    seed: int
    dataset_sha256: str


def train_from_npz(
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    layers: int = 2,
    dropout: float = 0.0,
    lambda_link: float = 0.2,
    lambda_outage: float = 0.1,
    objective: str = "full",
    link_config: LinkConfig | None = None,
    seed: int = 20260827,
) -> TrainingResult:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the ML extra with: pip install -e '.[ml]'") from exc

    from .torch_model import CommunicationAwareObjective, GRUTrajectoryPredictor

    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.load(dataset_path, allow_pickle=False)
    history = np.asarray(data["history_xy"], dtype=np.float32)
    future = np.asarray(data["future_xy"], dtype=np.float32)
    future_heading = (
        np.asarray(data["future_ego_heading_rad"], dtype=np.float32)
        if "future_ego_heading_rad" in data
        else np.zeros(future.shape[:2], dtype=np.float32)
    )
    scenario_id = np.asarray(data["scenario_id"]).astype(str)
    coordinate_frame = (
        str(np.asarray(data["coordinate_frame"]).item())
        if "coordinate_frame" in data
        else "legacy_unspecified"
    )
    if history.ndim != 3 or future.ndim != 3 or history.shape[-1] != 2:
        raise ValueError("Expected history_xy/future_xy with shape (samples, time, 2).")
    if "split" in data:
        split = np.asarray(data["split"]).astype(str)
        validation_mask = split == "development"
    else:
        unique = np.unique(scenario_id)
        rng = np.random.default_rng(seed)
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, int(np.ceil(0.15 * shuffled.size)))
        validation_scenarios = set(shuffled[-validation_count:])
        validation_mask = np.asarray(
            [item in validation_scenarios for item in scenario_id]
        )
    train_mask = ~validation_mask
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError(
            "Scenario-level split requires at least two distinct scenarios."
        )

    center = history[train_mask].mean(axis=(0, 1), keepdims=True)
    scale = history[train_mask].std(axis=(0, 1), keepdims=True)
    scale = np.maximum(scale, 1e-3)
    history_normalized = (history - center) / scale
    future_normalized = (future - center) / scale
    train_dataset = TensorDataset(
        torch.from_numpy(history_normalized[train_mask]),
        torch.from_numpy(future_normalized[train_mask]),
        torch.from_numpy(future_heading[train_mask]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(history_normalized[validation_mask]),
        torch.from_numpy(future_normalized[validation_mask]),
        torch.from_numpy(future_heading[validation_mask]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUTrajectoryPredictor(
        horizon_steps=future.shape[1],
        hidden_size=hidden_size,
        layers=layers,
        dropout=dropout,
    ).to(device)
    objective_weights = {
        "trajectory_only": (1.0, 0.0, 0.0),
        "trajectory_link": (1.0, lambda_link, 0.0),
        "trajectory_outage": (1.0, 0.0, lambda_outage),
        "full": (1.0, lambda_link, lambda_outage),
    }
    objective_name = objective
    if objective_name not in objective_weights:
        raise ValueError(
            f"Unknown objective {objective_name!r}; "
            f"choose from {sorted(objective_weights)}."
        )
    lambda_trajectory, effective_link, effective_outage = objective_weights[
        objective_name
    ]
    link = link_config or LinkConfig()
    objective_module = CommunicationAwareObjective(
        reference_distance_m=link.reference_distance_m,
        reference_snr_db=link.reference_snr_db,
        pointing_sigma_deg=link.pointing_sigma_deg,
        field_of_view_deg=link.field_of_view_deg,
        atmospheric_attenuation_per_m=link.atmospheric_attenuation_per_m,
        outage_ber_threshold=link.outage_ber_threshold,
        lambda_trajectory=lambda_trajectory,
        lambda_link=effective_link,
        lambda_outage=effective_outage,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    center_tensor = torch.as_tensor(center, device=device)
    scale_tensor = torch.as_tensor(scale, device=device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "best_comm_aware_gru.pt"
    best_loss = float("inf")
    best_epoch = -1
    patience = 12
    stale = 0

    for epoch in range(epochs):
        model.train()
        for history_batch, future_batch, heading_batch in train_loader:
            history_batch = history_batch.to(device)
            future_batch = future_batch.to(device)
            heading_batch = heading_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_normalized = model(history_batch)
            predicted_m = predicted_normalized * scale_tensor + center_tensor
            target_m = future_batch * scale_tensor + center_tensor
            loss, _ = objective_module(predicted_m, target_m, heading_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for history_batch, future_batch, heading_batch in validation_loader:
                history_batch = history_batch.to(device)
                future_batch = future_batch.to(device)
                predicted_normalized = model(history_batch)
                predicted_m = predicted_normalized * scale_tensor + center_tensor
                target_m = future_batch * scale_tensor + center_tensor
                loss, _ = objective_module(
                    predicted_m, target_m, heading_batch.to(device)
                )
                total_loss += float(loss)
                batches += 1
        validation_loss = total_loss / max(1, batches)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "horizon_steps": int(future.shape[1]),
                        "hidden_size": hidden_size,
                        "layers": layers,
                        "dropout": dropout,
                    },
                    "normalization": {"center": center, "scale": scale},
                    "feature_schema": {
                        "version": 1,
                        "input": "ego_relative_xy",
                        "input_axes": "world_xy",
                        "link_frame": coordinate_frame,
                        "future_ego_heading": "explicit_training_target_metadata",
                        "history_steps": int(history.shape[1]),
                        "horizon_steps": int(future.shape[1]),
                    },
                    "dataset": {
                        "file": Path(dataset_path).name,
                        "sha256": sha256_file(dataset_path),
                        "scenario_safe_split": True,
                    },
                    "training": {
                        "objective": objective_name,
                        "lambda_trajectory": lambda_trajectory,
                        "lambda_link": effective_link,
                        "lambda_outage": effective_outage,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "validation_loss": best_loss,
                    },
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    validation_breakdown = {"trajectory": 0.0, "link": 0.0, "outage": 0.0}
    validation_batches = 0
    with torch.no_grad():
        for history_batch, future_batch, heading_batch in validation_loader:
            predicted = model(history_batch.to(device))
            predicted = predicted * scale_tensor + center_tensor
            target = future_batch.to(device) * scale_tensor + center_tensor
            _, breakdown = objective_module(
                predicted, target, heading_batch.to(device)
            )
            for key in validation_breakdown:
                validation_breakdown[key] += float(breakdown[key])
            validation_batches += 1
            predictions.append(predicted.cpu().numpy())
            targets.append(target.cpu().numpy())
    error = np.linalg.norm(
        np.concatenate(predictions) - np.concatenate(targets), axis=-1
    )
    result = TrainingResult(
        checkpoint=str(checkpoint),
        best_epoch=best_epoch,
        validation_loss=best_loss,
        validation_ade_m=float(error.mean()),
        validation_fde_m=float(error[:, -1].mean()),
        validation_trajectory_loss=(
            validation_breakdown["trajectory"] / max(1, validation_batches)
        ),
        validation_link_loss=(
            validation_breakdown["link"] / max(1, validation_batches)
        ),
        validation_outage_loss=(
            validation_breakdown["outage"] / max(1, validation_batches)
        ),
        train_samples=int(train_mask.sum()),
        validation_samples=int(validation_mask.sum()),
        objective=str(saved["training"]["objective"]),
        seed=seed,
        dataset_sha256=str(saved["dataset"]["sha256"]),
    )
    (destination / "training_result.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )
    return result
