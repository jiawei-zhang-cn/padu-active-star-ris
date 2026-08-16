"""Strict run-configuration loading for the PADU submission package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .task_generation import ClosedInterval, TaskDistribution


def _exact_keys(
    mapping: dict[str, Any],
    required: set[str],
    location: str,
) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise ValueError(
            f"{location} keys are invalid; missing={missing}, unknown={unknown}"
        )


def _object(value: Any, location: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{location} must be set")
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _positive_int(value: Any, location: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{location} must be a positive integer")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{location} must be a positive integer")
    return number


def _nonnegative_int(value: Any, location: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{location} must be a non-negative integer")
    number = int(value)
    if number < 0 or number != value:
        raise ValueError(f"{location} must be a non-negative integer")
    return number


def _positive_float(value: Any, location: str) -> float:
    if value is None:
        raise ValueError(f"{location} must be set")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{location} must be finite and positive")
    return number


def _intervals(
    value: Any,
    location: str,
    number_of_users: int,
) -> tuple[ClosedInterval, ...]:
    if value is None:
        raise ValueError(f"{location} must be set")
    if not isinstance(value, list) or len(value) != number_of_users:
        raise ValueError(f"{location} must contain one interval per user")
    intervals = []
    for index, raw in enumerate(value):
        item_location = f"{location}[{index}]"
        mapping = _object(raw, item_location)
        _exact_keys(mapping, {"minimum", "maximum"}, item_location)
        intervals.append(
            ClosedInterval(
                float(mapping["minimum"]),
                float(mapping["maximum"]),
            )
        )
    return tuple(intervals)


def _strata(
    value: Any,
    location: str,
) -> tuple[ClosedInterval, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty list")
    intervals = []
    for index, raw in enumerate(value):
        item_location = f"{location}[{index}]"
        mapping = _object(raw, item_location)
        _exact_keys(mapping, {"minimum", "maximum"}, item_location)
        intervals.append(
            ClosedInterval(
                float(mapping["minimum"]),
                float(mapping["maximum"]),
            )
        )
    return tuple(intervals)


@dataclass(frozen=True)
class ArchitectureSettings:
    gru_hidden_size: int
    gru_number_of_layers: int
    warm_start_hidden_widths: tuple[int, ...]


@dataclass(frozen=True)
class DatasetSettings:
    trajectory_slots: int
    gru_training_tasks: int
    gru_validation_tasks: int
    cold_start_tasks: int
    meta_training_tasks: int
    meta_validation_tasks: int
    in_domain_test_tasks: int
    out_of_domain_test_tasks: int
    meta_support_size: int
    meta_query_size: int


@dataclass(frozen=True)
class PhysicsLossSettings:
    feasible_mapping_epsilon: float
    qos_shortfall_penalty_weight: float


@dataclass(frozen=True)
class EvaluationSettings:
    schemes: tuple[str, ...]
    reference_ao_iterations: int
    random_raw_standard_deviation: float


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    root_seeds: tuple[int, ...]
    require_cuda: bool
    architecture: ArchitectureSettings
    dataset: DatasetSettings
    physics_loss: PhysicsLossSettings
    evaluation: EvaluationSettings
    task_distributions: dict[str, TaskDistribution]


ROOT_KEYS = {
    "schema_version",
    "root_seeds",
    "require_cuda",
    "architecture",
    "dataset",
    "physics_loss",
    "evaluation",
    "task_distributions",
}


ARCHITECTURE_KEYS = {
    "gru_hidden_size",
    "gru_number_of_layers",
    "warm_start_hidden_widths",
}


DATASET_KEYS = {
    "trajectory_slots",
    "gru_training_tasks",
    "gru_validation_tasks",
    "cold_start_tasks",
    "meta_training_tasks",
    "meta_validation_tasks",
    "in_domain_test_tasks",
    "out_of_domain_test_tasks",
    "meta_support_size",
    "meta_query_size",
}


PHYSICS_LOSS_KEYS = {
    "feasible_mapping_epsilon",
    "qos_shortfall_penalty_weight",
}


EVALUATION_KEYS = {
    "schemes",
    "reference_ao_iterations",
    "random_raw_standard_deviation",
}


TASK_DISTRIBUTION_KEYS = {
    "user_speed_intervals_m_per_s",
    "speed_strata_m_per_s",
    "rician_factor_strata_db",
    "user_rician_factor_intervals_db",
}


def load_run_config(
    path: str | Path,
    *,
    number_of_users: int,
    history_length: int,
) -> RunConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run configuration root must be an object")
    _exact_keys(raw, ROOT_KEYS, "root")
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must equal 1")
    if number_of_users <= 0:
        raise ValueError("number_of_users must be positive")
    if history_length <= 0:
        raise ValueError("history_length must be positive")

    root_seeds = tuple(
        _nonnegative_int(value, "root_seeds") for value in raw["root_seeds"]
    )
    if not root_seeds:
        raise ValueError("root_seeds must be non-empty")

    architecture_raw = _object(raw["architecture"], "architecture")
    _exact_keys(architecture_raw, ARCHITECTURE_KEYS, "architecture")
    widths = tuple(
        _positive_int(value, "architecture.warm_start_hidden_widths")
        for value in architecture_raw["warm_start_hidden_widths"]
    )
    if not widths:
        raise ValueError("architecture.warm_start_hidden_widths must be non-empty")
    architecture = ArchitectureSettings(
        gru_hidden_size=_positive_int(
            architecture_raw["gru_hidden_size"],
            "architecture.gru_hidden_size",
        ),
        gru_number_of_layers=_positive_int(
            architecture_raw["gru_number_of_layers"],
            "architecture.gru_number_of_layers",
        ),
        warm_start_hidden_widths=widths,
    )

    dataset_raw = _object(raw["dataset"], "dataset")
    _exact_keys(dataset_raw, DATASET_KEYS, "dataset")
    dataset = DatasetSettings(
        trajectory_slots=_positive_int(
            dataset_raw["trajectory_slots"], "dataset.trajectory_slots"
        ),
        gru_training_tasks=_positive_int(
            dataset_raw["gru_training_tasks"], "dataset.gru_training_tasks"
        ),
        gru_validation_tasks=_positive_int(
            dataset_raw["gru_validation_tasks"], "dataset.gru_validation_tasks"
        ),
        cold_start_tasks=_positive_int(
            dataset_raw["cold_start_tasks"], "dataset.cold_start_tasks"
        ),
        meta_training_tasks=_positive_int(
            dataset_raw["meta_training_tasks"], "dataset.meta_training_tasks"
        ),
        meta_validation_tasks=_positive_int(
            dataset_raw["meta_validation_tasks"], "dataset.meta_validation_tasks"
        ),
        in_domain_test_tasks=_positive_int(
            dataset_raw["in_domain_test_tasks"], "dataset.in_domain_test_tasks"
        ),
        out_of_domain_test_tasks=_positive_int(
            dataset_raw["out_of_domain_test_tasks"],
            "dataset.out_of_domain_test_tasks",
        ),
        meta_support_size=_positive_int(
            dataset_raw["meta_support_size"], "dataset.meta_support_size"
        ),
        meta_query_size=_positive_int(
            dataset_raw["meta_query_size"], "dataset.meta_query_size"
        ),
    )

    physics_loss_raw = _object(raw["physics_loss"], "physics_loss")
    _exact_keys(physics_loss_raw, PHYSICS_LOSS_KEYS, "physics_loss")
    physics_loss = PhysicsLossSettings(
        feasible_mapping_epsilon=_positive_float(
            physics_loss_raw["feasible_mapping_epsilon"],
            "physics_loss.feasible_mapping_epsilon",
        ),
        qos_shortfall_penalty_weight=_positive_float(
            physics_loss_raw["qos_shortfall_penalty_weight"],
            "physics_loss.qos_shortfall_penalty_weight",
        ),
    )

    evaluation_raw = _object(raw["evaluation"], "evaluation")
    _exact_keys(evaluation_raw, EVALUATION_KEYS, "evaluation")
    schemes = tuple(str(value) for value in evaluation_raw["schemes"])
    if not schemes:
        raise ValueError("evaluation.schemes must be non-empty")
    evaluation = EvaluationSettings(
        schemes=schemes,
        reference_ao_iterations=_positive_int(
            evaluation_raw["reference_ao_iterations"],
            "evaluation.reference_ao_iterations",
        ),
        random_raw_standard_deviation=_positive_float(
            evaluation_raw["random_raw_standard_deviation"],
            "evaluation.random_raw_standard_deviation",
        ),
    )

    task_distributions_raw = _object(
        raw["task_distributions"], "task_distributions"
    )
    task_distributions: dict[str, TaskDistribution] = {}
    for name, value in task_distributions_raw.items():
        location = f"task_distributions.{name}"
        mapping = _object(value, location)
        unknown = sorted(set(mapping) - TASK_DISTRIBUTION_KEYS)
        if unknown:
            raise ValueError(
                f"{location} keys are invalid; missing=[], unknown={unknown}"
            )
        task_distributions[name] = TaskDistribution(
            user_speed_intervals_m_per_s=_intervals(
                mapping.get("user_speed_intervals_m_per_s"),
                f"{location}.user_speed_intervals_m_per_s",
                number_of_users,
            ),
            user_rician_factor_intervals_db=_intervals(
                mapping.get("user_rician_factor_intervals_db"),
                f"{location}.user_rician_factor_intervals_db",
                number_of_users,
            ),
            speed_strata_m_per_s=_strata(
                mapping.get("speed_strata_m_per_s"),
                f"{location}.speed_strata_m_per_s",
            ),
            rician_factor_strata_db=_strata(
                mapping.get("rician_factor_strata_db"),
                f"{location}.rician_factor_strata_db",
            ),
        )

    return RunConfig(
        schema_version=1,
        root_seeds=root_seeds,
        require_cuda=bool(raw["require_cuda"]),
        architecture=architecture,
        dataset=dataset,
        physics_loss=physics_loss,
        evaluation=evaluation,
        task_distributions=task_distributions,
    )
