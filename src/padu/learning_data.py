"""Minimal trajectory records used by the PADU data generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex64]


@dataclass(frozen=True)
class ChannelTrajectoryRecord:
    trajectory_id: str
    channels: ComplexArray

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        channels = np.asarray(self.channels, dtype=np.complex64)
        if channels.ndim != 3:
            raise ValueError("channels must have shape (time, elements, users)")
        if channels.shape[0] < 2 or channels.shape[1] < 1 or channels.shape[2] < 1:
            raise ValueError("channels must contain time, elements, and users")
        if not np.all(np.isfinite(channels)):
            raise ValueError("channels must be finite")
        object.__setattr__(self, "channels", channels.copy())
