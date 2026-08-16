"""Continuous-time random waypoint mobility sampled at slot boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
WaypointSampler = Callable[[], FloatArray]


@dataclass(frozen=True)
class Rectangle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(np.isfinite(values)):
            raise ValueError("rectangle bounds must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("rectangle bounds must define a non-empty area")

    def contains(self, point_xy: FloatArray, tolerance: float = 0.0) -> bool:
        point = _point_xy(point_xy, "point_xy")
        return bool(
            self.x_min - tolerance <= point[0] <= self.x_max + tolerance
            and self.y_min - tolerance <= point[1] <= self.y_max + tolerance
        )

    def sample(self, rng: np.random.Generator) -> FloatArray:
        return np.array(
            [
                rng.uniform(self.x_min, self.x_max),
                rng.uniform(self.y_min, self.y_max),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class WaypointState:
    position_xy_m: FloatArray
    waypoint_xy_m: FloatArray
    speed_m_per_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_xy_m",
            _point_xy(self.position_xy_m, "position_xy_m"),
        )
        object.__setattr__(
            self,
            "waypoint_xy_m",
            _point_xy(self.waypoint_xy_m, "waypoint_xy_m"),
        )
        if not np.isfinite(self.speed_m_per_s) or self.speed_m_per_s < 0.0:
            raise ValueError("speed_m_per_s must be finite and non-negative")


def _point_xy(value: FloatArray, name: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite two-dimensional point")
    return point.copy()


def advance_random_waypoint(
    state: WaypointState,
    slot_duration_s: float,
    arrival_tolerance_m: float,
    sample_waypoint: WaypointSampler,
    maximum_waypoint_transitions: int = 100_000,
) -> WaypointState:
    """Advance one slot and immediately use time left after reaching a waypoint."""
    if slot_duration_s <= 0.0 or not np.isfinite(slot_duration_s):
        raise ValueError("slot_duration_s must be finite and positive")
    if arrival_tolerance_m <= 0.0 or not np.isfinite(arrival_tolerance_m):
        raise ValueError("arrival_tolerance_m must be finite and positive")
    if maximum_waypoint_transitions <= 0:
        raise ValueError("maximum_waypoint_transitions must be positive")

    position = state.position_xy_m.copy()
    waypoint = state.waypoint_xy_m.copy()
    remaining_distance = state.speed_m_per_s * slot_duration_s
    transitions = 0

    while remaining_distance > arrival_tolerance_m:
        displacement = waypoint - position
        distance = float(np.linalg.norm(displacement))

        if distance <= arrival_tolerance_m:
            waypoint = _point_xy(sample_waypoint(), "sampled waypoint")
            transitions += 1
            if transitions > maximum_waypoint_transitions:
                raise RuntimeError("too many waypoint transitions in one slot")
            continue

        if distance > remaining_distance:
            position += displacement * (remaining_distance / distance)
            remaining_distance = 0.0
        else:
            position = waypoint.copy()
            remaining_distance -= distance
            if remaining_distance > arrival_tolerance_m:
                waypoint = _point_xy(sample_waypoint(), "sampled waypoint")
                transitions += 1
                if transitions > maximum_waypoint_transitions:
                    raise RuntimeError("too many waypoint transitions in one slot")

    return WaypointState(
        position_xy_m=position,
        waypoint_xy_m=waypoint,
        speed_m_per_s=state.speed_m_per_s,
    )


def initialize_random_waypoint(
    region: Rectangle,
    speed_m_per_s: float,
    arrival_tolerance_m: float,
    rng: np.random.Generator,
) -> WaypointState:
    if arrival_tolerance_m <= 0.0:
        raise ValueError("arrival_tolerance_m must be positive")
    position = region.sample(rng)
    waypoint = region.sample(rng)
    while np.linalg.norm(waypoint - position) <= arrival_tolerance_m:
        waypoint = region.sample(rng)
    return WaypointState(position, waypoint, speed_m_per_s)


def simulate_random_waypoint(
    region: Rectangle,
    speed_m_per_s: float,
    slot_duration_s: float,
    number_of_slots: int,
    arrival_tolerance_m: float,
    rng: np.random.Generator,
) -> FloatArray:
    if number_of_slots < 0:
        raise ValueError("number_of_slots must be non-negative")
    state = initialize_random_waypoint(
        region,
        speed_m_per_s,
        arrival_tolerance_m,
        rng,
    )
    positions = np.empty((number_of_slots + 1, 2), dtype=np.float64)
    positions[0] = state.position_xy_m
    for slot in range(number_of_slots):
        state = advance_random_waypoint(
            state,
            slot_duration_s,
            arrival_tolerance_m,
            lambda: region.sample(rng),
        )
        if not region.contains(state.position_xy_m, tolerance=arrival_tolerance_m):
            raise RuntimeError("random waypoint position left its convex region")
        positions[slot + 1] = state.position_xy_m
    return positions

