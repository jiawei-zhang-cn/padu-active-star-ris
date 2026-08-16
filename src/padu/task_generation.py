"""Meta-task parameter specifications and reproducible trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ExperimentConfig
from .learning_data import ChannelTrajectoryRecord
from .simulation import (
    FixedDeployment,
    MobileTrajectory,
    generate_mobile_trajectory,
)


@dataclass(frozen=True)
class ClosedInterval:
    minimum: float
    maximum: float

    def validate(self, name: str, nonnegative: bool) -> None:
        values = (self.minimum, self.maximum)
        if any(not np.isfinite(value) for value in values):
            raise ValueError(f"{name} endpoints must be finite")
        if self.minimum > self.maximum:
            raise ValueError(f"{name}.minimum must not exceed maximum")
        if nonnegative and self.minimum < 0.0:
            raise ValueError(f"{name}.minimum must be non-negative")


@dataclass(frozen=True)
class TaskDistribution:
    """Per-user uniform distributions with optional joint task strata."""

    user_speed_intervals_m_per_s: tuple[ClosedInterval, ...]
    user_rician_factor_intervals_db: tuple[ClosedInterval, ...]
    speed_strata_m_per_s: tuple[ClosedInterval, ...] | None = None
    rician_factor_strata_db: tuple[ClosedInterval, ...] | None = None

    def validate(self, number_of_users: int) -> None:
        if number_of_users <= 0:
            raise ValueError("number_of_users must be positive")
        if len(self.user_speed_intervals_m_per_s) != number_of_users:
            raise ValueError(
                "one user speed interval is required for each user"
            )
        if len(self.user_rician_factor_intervals_db) != number_of_users:
            raise ValueError(
                "one Rician-factor interval is required for each user"
            )
        for index, interval in enumerate(
            self.user_speed_intervals_m_per_s
        ):
            interval.validate(
                f"user_speed_intervals_m_per_s[{index}]",
                nonnegative=True,
            )
        for index, interval in enumerate(
            self.user_rician_factor_intervals_db
        ):
            interval.validate(
                f"user_rician_factor_intervals_db[{index}]",
                nonnegative=False,
            )
        if self.speed_strata_m_per_s is not None:
            if not self.speed_strata_m_per_s:
                raise ValueError("speed_strata_m_per_s must not be empty")
            for index, interval in enumerate(self.speed_strata_m_per_s):
                interval.validate(
                    f"speed_strata_m_per_s[{index}]",
                    nonnegative=True,
                )
        if self.rician_factor_strata_db is not None:
            if not self.rician_factor_strata_db:
                raise ValueError("rician_factor_strata_db must not be empty")
            for index, interval in enumerate(
                self.rician_factor_strata_db
            ):
                interval.validate(
                    f"rician_factor_strata_db[{index}]",
                    nonnegative=False,
                )

    def sample(
        self,
        rng: np.random.Generator,
        stratum_index: int | None = None,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        speed_intervals = self.user_speed_intervals_m_per_s
        rician_intervals = self.user_rician_factor_intervals_db
        if stratum_index is not None:
            speed_stratum_index = stratum_index
            rician_stratum_index = stratum_index
            if (
                self.speed_strata_m_per_s is not None
                and self.rician_factor_strata_db is not None
            ):
                speed_stratum_index = (
                    stratum_index // len(self.rician_factor_strata_db)
                )
                rician_stratum_index = stratum_index
            if self.speed_strata_m_per_s is not None:
                speed_stratum = self.speed_strata_m_per_s[
                    speed_stratum_index % len(self.speed_strata_m_per_s)
                ]
                speed_intervals = tuple(
                    speed_stratum
                    for _ in self.user_speed_intervals_m_per_s
                )
            if self.rician_factor_strata_db is not None:
                rician_stratum = self.rician_factor_strata_db[
                    rician_stratum_index
                    % len(self.rician_factor_strata_db)
                ]
                rician_intervals = tuple(
                    rician_stratum
                    for _ in self.user_rician_factor_intervals_db
                )
        speeds = tuple(
            float(rng.uniform(interval.minimum, interval.maximum))
            if interval.minimum < interval.maximum
            else float(interval.minimum)
            for interval in speed_intervals
        )
        rician_factors = tuple(
            float(rng.uniform(interval.minimum, interval.maximum))
            if interval.minimum < interval.maximum
            else float(interval.minimum)
            for interval in rician_intervals
        )
        return speeds, rician_factors


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    trajectory: MobileTrajectory

    @property
    def channel_record(self) -> ChannelTrajectoryRecord:
        return ChannelTrajectoryRecord(
            trajectory_id=self.task_id,
            channels=self.trajectory.surface_to_user_channels,
        )


def generate_task_family(
    *,
    config: ExperimentConfig,
    deployment: FixedDeployment,
    distribution: TaskDistribution,
    number_of_tasks: int,
    number_of_slots: int,
    task_id_prefix: str,
    rng: np.random.Generator,
) -> tuple[GeneratedTask, ...]:
    """Generate isolated trajectories while reusing one fixed deployment."""
    distribution.validate(config.number_of_users)
    if number_of_tasks <= 0:
        raise ValueError("number_of_tasks must be positive")
    if number_of_slots <= 0:
        raise ValueError("number_of_slots must be positive")
    if not task_id_prefix:
        raise ValueError("task_id_prefix must be non-empty")

    tasks = []
    for task_index in range(number_of_tasks):
        speeds, rician_factors = distribution.sample(
            rng,
            stratum_index=(
                task_index
                if (
                    distribution.speed_strata_m_per_s is not None
                    or distribution.rician_factor_strata_db is not None
                )
                else None
            ),
        )
        trajectory = generate_mobile_trajectory(
            config=config,
            deployment=deployment,
            number_of_slots=number_of_slots,
            rng=rng,
            user_speeds_m_per_s=speeds,
            user_rician_factor_db=rician_factors,
        )
        tasks.append(
            GeneratedTask(
                task_id=f"{task_id_prefix}-{task_index:05d}",
                trajectory=trajectory,
            )
        )
    return tuple(tasks)


def assert_disjoint_task_ids(
    *task_families: Sequence[GeneratedTask],
) -> None:
    seen: set[str] = set()
    for family in task_families:
        identifiers = [task.task_id for task in family]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("task identifiers must be unique within a family")
        overlap = seen.intersection(identifiers)
        if overlap:
            raise ValueError(
                f"task families contain duplicate identifiers: {sorted(overlap)}"
            )
        seen.update(identifiers)
