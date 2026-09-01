from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TorchCheckpointPredictor:
    """Inference adapter for a checkpoint produced by ``pcfmcw train``."""

    name = "communication_aware_gru"

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            message = "Install the ML extra with: pip install -e '.[ml]'"
            raise ImportError(message) from exc
        from .torch_model import GRUTrajectoryPredictor

        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        feature_schema = payload.get("feature_schema")
        if not isinstance(feature_schema, dict) or feature_schema.get("version") != 1:
            raise ValueError(
                "Checkpoint lacks the version-1 local feature schema; upstream "
                "Stage-4 reports/checkpoints are not silently compatible."
            )
        if feature_schema.get("input") != "ego_relative_xy":
            raise ValueError("Checkpoint input schema is not ego_relative_xy.")
        self.feature_schema = dict(feature_schema)
        model_config = payload["model_config"]
        self.horizon_steps = int(model_config["horizon_steps"])
        self.model = GRUTrajectoryPredictor(**model_config).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.center = np.asarray(payload["normalization"]["center"], dtype=np.float32)
        self.scale = np.asarray(payload["normalization"]["scale"], dtype=np.float32)

    def predict(
        self, history_xy: ArrayLike, horizon_steps: int, dt_s: float
    ) -> NDArray[np.float64]:
        del dt_s
        if horizon_steps > self.horizon_steps:
            message = (
                f"Checkpoint horizon is {self.horizon_steps}, "
                f"requested {horizon_steps}."
            )
            raise ValueError(
                message
            )
        history = np.asarray(history_xy, dtype=np.float32)
        if history.ndim != 3 or history.shape[-1] != 2:
            raise ValueError("Learned predictor expects shape (actors, history, 2).")
        expected_history = int(self.feature_schema["history_steps"])
        if history.shape[1] != expected_history:
            raise ValueError(
                f"Checkpoint history is {expected_history} steps, received "
                f"{history.shape[1]}."
            )
        normalized = (history - self.center) / self.scale
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(normalized).to(self.device)
            output = self.model(tensor).cpu().numpy()[:, :horizon_steps]
        return (output * self.scale + self.center).astype(np.float64)
