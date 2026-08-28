from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: str
    best_epoch: int
    validation_loss: float
    validation_ade_m: float
    validation_fde_m: float
    train_samples: int
    validation_samples: int


def train_from_npz(
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    layers: int = 2,
    lambda_link: float = 0.2,
    lambda_outage: float = 0.1,
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
    scenario_id = np.asarray(data["scenario_id"]).astype(str)
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
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(history_normalized[validation_mask]),
        torch.from_numpy(future_normalized[validation_mask]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUTrajectoryPredictor(
        horizon_steps=future.shape[1], hidden_size=hidden_size, layers=layers
    ).to(device)
    objective = CommunicationAwareObjective(
        lambda_link=lambda_link, lambda_outage=lambda_outage
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
        for history_batch, future_batch in train_loader:
            history_batch = history_batch.to(device)
            future_batch = future_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_normalized = model(history_batch)
            predicted_m = predicted_normalized * scale_tensor + center_tensor
            target_m = future_batch * scale_tensor + center_tensor
            loss, _ = objective(predicted_m, target_m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for history_batch, future_batch in validation_loader:
                history_batch = history_batch.to(device)
                future_batch = future_batch.to(device)
                predicted_normalized = model(history_batch)
                predicted_m = predicted_normalized * scale_tensor + center_tensor
                target_m = future_batch * scale_tensor + center_tensor
                loss, _ = objective(predicted_m, target_m)
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
                    },
                    "normalization": {"center": center, "scale": scale},
                    "training": {
                        "lambda_link": lambda_link,
                        "lambda_outage": lambda_outage,
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
    with torch.no_grad():
        for history_batch, future_batch in validation_loader:
            predicted = model(history_batch.to(device))
            predicted = predicted * scale_tensor + center_tensor
            target = future_batch.to(device) * scale_tensor + center_tensor
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
        train_samples=int(train_mask.sum()),
        validation_samples=int(validation_mask.sum()),
    )
    (destination / "training_result.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )
    return result
