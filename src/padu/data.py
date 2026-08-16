"""Leakage-safe channel trajectory preprocessing and datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


ComplexArray = NDArray[np.complex64]


def complex_to_real_features(channel: ComplexArray) -> NDArray[np.float32]:
    value = np.asarray(channel, dtype=np.complex64)
    return np.concatenate([value.real, value.imag], axis=-1).astype(np.float32)


def real_features_to_complex(features: NDArray[np.float32]) -> ComplexArray:
    value = np.asarray(features, dtype=np.float32)
    if value.shape[-1] % 2 != 0:
        raise ValueError("the last feature dimension must be even")
    elements = value.shape[-1] // 2
    return (
        value[..., :elements] + 1j * value[..., elements:]
    ).astype(np.complex64)


@dataclass(frozen=True)
class ComplexFeatureNormalizer:
    mean: NDArray[np.float32]
    standard_deviation: NDArray[np.float32]

    @classmethod
    def fit(
        cls,
        training_trajectories: Sequence[ComplexArray],
        minimum_standard_deviation: float = 1.0e-8,
    ) -> "ComplexFeatureNormalizer":
        if not training_trajectories:
            raise ValueError("training_trajectories must be non-empty")
        feature_blocks = []
        reference_shape = None
        for trajectory in training_trajectories:
            value = np.asarray(trajectory, dtype=np.complex64)
            if value.ndim != 3:
                raise ValueError(
                    "each trajectory must have shape (time, users, elements)"
                )
            if reference_shape is None:
                reference_shape = value.shape[1:]
            elif value.shape[1:] != reference_shape:
                raise ValueError("all trajectories must have equal user/element shape")
            feature_blocks.append(
                complex_to_real_features(value).reshape(-1, 2 * value.shape[-1])
            )
        samples = np.concatenate(feature_blocks, axis=0)
        mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
        standard_deviation = samples.std(axis=0, dtype=np.float64).astype(np.float32)
        standard_deviation = np.maximum(
            standard_deviation,
            np.float32(minimum_standard_deviation),
        )
        return cls(mean=mean, standard_deviation=standard_deviation)

    def transform(self, channel: ComplexArray) -> NDArray[np.float32]:
        features = complex_to_real_features(channel)
        if features.shape[-1] != self.mean.size:
            raise ValueError("channel element count does not match the normalizer")
        return ((features - self.mean) / self.standard_deviation).astype(np.float32)

    def inverse_transform(
        self,
        normalized_features: NDArray[np.float32],
    ) -> ComplexArray:
        features = np.asarray(normalized_features, dtype=np.float32)
        if features.shape[-1] != self.mean.size:
            raise ValueError("feature count does not match the normalizer")
        restored = features * self.standard_deviation + self.mean
        return real_features_to_complex(restored)


def split_trajectory_indices(
    number_of_trajectories: int,
    training_count: int,
    validation_count: int,
    test_count: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.int64]]:
    counts = (training_count, validation_count, test_count)
    if number_of_trajectories <= 0:
        raise ValueError("number_of_trajectories must be positive")
    if any(count <= 0 for count in counts):
        raise ValueError("all split counts must be positive")
    if sum(counts) != number_of_trajectories:
        raise ValueError("split counts must sum to number_of_trajectories")
    order = rng.permutation(number_of_trajectories)
    train_end = training_count
    validation_end = train_end + validation_count
    return {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "test": order[validation_end:],
    }


class SharedGRUWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Each sample is one user window from one complete trajectory split."""

    def __init__(
        self,
        trajectories: Sequence[ComplexArray],
        normalizer: ComplexFeatureNormalizer,
        history_length: int,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self._windows: list[Tensor] = []
        self._targets: list[Tensor] = []
        for trajectory in trajectories:
            value = np.asarray(trajectory, dtype=np.complex64)
            if value.ndim != 3:
                raise ValueError(
                    "each trajectory must have shape (time, users, elements)"
                )
            if value.shape[0] <= history_length:
                raise ValueError(
                    "each trajectory must be longer than history_length"
                )
            normalized = normalizer.transform(value)
            for time_index in range(history_length - 1, value.shape[0] - 1):
                for user_index in range(value.shape[1]):
                    self._windows.append(
                        torch.from_numpy(
                            normalized[
                                time_index - history_length + 1 : time_index + 1,
                                user_index,
                            ].copy()
                        )
                    )
                    self._targets.append(
                        torch.from_numpy(
                            normalized[time_index + 1, user_index].copy()
                        )
                    )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._windows[index], self._targets[index]


def select_complete_trajectories(
    trajectories: Sequence[ComplexArray],
    indices: Iterable[int],
) -> list[ComplexArray]:
    return [np.asarray(trajectories[index], dtype=np.complex64) for index in indices]
