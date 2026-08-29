from __future__ import annotations

import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ImportError as exc:  # pragma: no cover - exercised only without ML extra
    raise ImportError(
        "PyTorch is required for communication-aware training. "
        "Install the project with: pip install -e '.[ml]'"
    ) from exc


class GRUTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        horizon_steps: int,
        hidden_size: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.encoder = nn.GRU(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * horizon_steps),
        )

    def forward(self, history_xy: torch.Tensor) -> torch.Tensor:
        velocity = torch.diff(history_xy, dim=1, prepend=history_xy[:, :1])
        features = torch.cat([history_xy, velocity], dim=-1)
        _, hidden = self.encoder(features)
        offsets = self.head(hidden[-1]).reshape(-1, self.horizon_steps, 2)
        return history_xy[:, -1:, :] + offsets


class CommunicationAwareObjective(nn.Module):
    def __init__(
        self,
        reference_distance_m: float = 40.0,
        reference_snr_db: float = 18.0,
        pointing_sigma_deg: float = 18.0,
        field_of_view_deg: float = 70.0,
        atmospheric_attenuation_per_m: float = 0.004,
        outage_ber_threshold: float = 1e-3,
        lambda_link: float = 0.2,
        lambda_outage: float = 0.1,
        lambda_trajectory: float = 1.0,
        outage_temperature: float = 0.5,
        fov_softness_deg: float = 1.0,
    ):
        super().__init__()
        self.reference_distance_m = reference_distance_m
        self.reference_snr_db = reference_snr_db
        self.pointing_sigma_rad = math.radians(pointing_sigma_deg)
        self.fov_half_rad = math.radians(field_of_view_deg / 2)
        self.fov_softness_rad = math.radians(fov_softness_deg)
        self.atmospheric = atmospheric_attenuation_per_m
        self.outage_ber_threshold = outage_ber_threshold
        self.lambda_link = lambda_link
        self.lambda_outage = lambda_outage
        self.lambda_trajectory = lambda_trajectory
        self.outage_temperature = outage_temperature

    def _bearing(
        self,
        relative_xy: torch.Tensor,
        ego_heading_rad: torch.Tensor | float,
    ) -> torch.Tensor:
        raw = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
        delta = raw - torch.as_tensor(
            ego_heading_rad, dtype=relative_xy.dtype, device=relative_xy.device
        )
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _log_snr(
        self,
        relative_xy: torch.Tensor,
        ego_heading_rad: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        distance = torch.linalg.vector_norm(relative_xy, dim=-1).clamp_min(0.5)
        bearing = self._bearing(relative_xy, ego_heading_rad)
        fov_transmission = torch.sigmoid(
            (self.fov_half_rad - torch.abs(bearing))
            / self.fov_softness_rad
        ).clamp_min(1e-12)
        log_gain = (
            -2 * torch.log(distance)
            - self.atmospheric * distance
            - 0.5 * (bearing / self.pointing_sigma_rad) ** 2
            + torch.log(fov_transmission)
        )
        reference_log_gain = (
            -2 * math.log(self.reference_distance_m)
            - self.atmospheric * self.reference_distance_m
        )
        return (
            math.log(10) * self.reference_snr_db / 10
            + log_gain
            - reference_log_gain
        )

    def forward(
        self,
        predicted_xy: torch.Tensor,
        target_xy: torch.Tensor,
        ego_heading_rad: torch.Tensor | float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        trajectory = functional.smooth_l1_loss(predicted_xy, target_xy)
        predicted_log_snr = self._log_snr(predicted_xy, ego_heading_rad)
        target_log_snr = self._log_snr(target_xy, ego_heading_rad)
        link = functional.smooth_l1_loss(predicted_log_snr, target_log_snr)
        gamma_threshold = -math.log(2 * self.outage_ber_threshold)
        target_bearing = self._bearing(target_xy, ego_heading_rad)
        target_outage = (
            (target_log_snr < math.log(gamma_threshold))
            | (torch.abs(target_bearing) > self.fov_half_rad)
        ).float()
        logits = (
            math.log(gamma_threshold) - predicted_log_snr
        ) / self.outage_temperature
        outage = functional.binary_cross_entropy_with_logits(logits, target_outage)
        total = (
            self.lambda_trajectory * trajectory
            + self.lambda_link * link
            + self.lambda_outage * outage
        )
        return total, {"trajectory": trajectory, "link": link, "outage": outage}
