"""Shared GRU predictor and real-valued warm-start MLP."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class SharedGRUPredictor(nn.Module):
    def __init__(
        self,
        number_of_surface_elements: int,
        hidden_size: int,
        number_of_layers: int,
    ) -> None:
        super().__init__()
        if number_of_surface_elements <= 0:
            raise ValueError("number_of_surface_elements must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be positive")
        feature_size = 2 * number_of_surface_elements
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, feature_size)

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape (batch, history, 2N)")
        states, _ = self.gru(history)
        return self.output_layer(states[:, -1, :])


class IncrementalSharedGRUPredictor(nn.Module):
    """Shared per-user GRU that predicts a normalized channel increment."""

    def __init__(
        self,
        number_of_surface_elements: int,
        hidden_size: int,
        number_of_layers: int,
        input_mode: str = "channel_only",
        predict_distribution_scale: bool = False,
    ) -> None:
        super().__init__()
        if number_of_surface_elements <= 0:
            raise ValueError("number_of_surface_elements must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be positive")
        if input_mode not in {
            "channel_only",
            "channel_and_first_difference",
        }:
            raise ValueError("input_mode is unsupported")
        feature_size = 2 * number_of_surface_elements
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.input_mode = input_mode
        self.predict_distribution_scale = predict_distribution_scale
        self.gru = nn.GRU(
            input_size=(
                feature_size
                if input_mode == "channel_only"
                else 2 * feature_size
            ),
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
        )
        self.increment_layer = nn.Linear(hidden_size, feature_size)
        self.scale_layer = (
            nn.Linear(hidden_size, 1)
            if predict_distribution_scale
            else None
        )

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        if history.ndim != 3:
            raise ValueError("history must have shape (batch, history, 2N)")
        expected_feature_size = (
            self.feature_size
            if self.input_mode == "channel_only"
            else 2 * self.feature_size
        )
        if history.shape[-1] != expected_feature_size:
            raise ValueError(
                "history feature dimension does not match input_mode"
            )
        states, _ = self.gru(history)
        final_state = states[:, -1, :]
        latest_channel = history[:, -1, : self.feature_size]
        normalized_prediction = (
            latest_channel + self.increment_layer(final_state)
        )
        return normalized_prediction, final_state

    def distribution(
        self, history: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized_prediction, final_state = self(history)
        if self.scale_layer is None:
            raise RuntimeError("predictor distribution scale head is missing")
        normalized_standard_deviation = (
            torch.nn.functional.softplus(
                self.scale_layer(final_state).squeeze(-1)
            )
            + 1.0e-4
        )
        return (
            normalized_prediction,
            final_state,
            normalized_standard_deviation,
        )


class WarmStartMLP(nn.Module):
    def __init__(
        self,
        number_of_bs_antennas: int,
        number_of_surface_elements: int,
        number_of_users: int,
        hidden_widths: Sequence[int],
        surface_architecture: str = "active",
    ) -> None:
        super().__init__()
        dimensions = (
            number_of_bs_antennas,
            number_of_surface_elements,
            number_of_users,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("M, N, and K must be positive")
        if not hidden_widths or any(width <= 0 for width in hidden_widths):
            raise ValueError("hidden_widths must contain positive integers")
        if surface_architecture not in {"active", "ideal_passive"}:
            raise ValueError("unsupported surface_architecture")

        self.number_of_bs_antennas = number_of_bs_antennas
        self.number_of_surface_elements = number_of_surface_elements
        self.number_of_users = number_of_users
        self.surface_architecture = surface_architecture
        self.input_size = (
            2 * number_of_surface_elements * number_of_users + number_of_users
        )
        surface_output_size = (
            4 * number_of_surface_elements
            if surface_architecture == "active"
            else 3 * number_of_surface_elements
        )
        self.output_size = (
            2 * number_of_bs_antennas * number_of_users
            + surface_output_size
        )

        layers: list[nn.Module] = []
        previous_width = self.input_size
        for width in hidden_widths:
            layers.extend([nn.Linear(previous_width, width), nn.ReLU()])
            previous_width = width
        layers.append(nn.Linear(previous_width, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_size:
            raise ValueError(
                f"features last dimension must equal {self.input_size}"
            )
        return self.network(features)
