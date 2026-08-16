"""Small runtime helpers shared by PADU training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audit import ExperimentAudit, audit_experiment
from .config import ExperimentConfig, load_experiment_config
from .run_config import RunConfig, load_run_config
from .task_generation import GeneratedTask


SEED_STREAM_NAMES = (
    "deployment",
    "gru_training_tasks",
    "gru_validation_tasks",
    "cold_start_tasks",
    "meta_training_tasks",
    "meta_validation_tasks",
    "in_domain_test_tasks",
    "out_of_domain_test_tasks",
    "gru_model_initialization",
    "gru_training",
    "cold_start_sampling",
    "warm_start_model_initialization",
    "meta_training",
    "passive_model_initialization",
    "passive_meta_training",
    "joint_model_initialization",
    "joint_training",
    "in_domain_execution",
    "out_of_domain_execution",
)


@dataclass(frozen=True)
class SeedManifest:
    root_seed: int
    streams: dict[str, int]


def derive_seed_manifest(root_seed: int) -> SeedManifest:
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    children = np.random.SeedSequence(root_seed).spawn(
        len(SEED_STREAM_NAMES)
    )
    streams = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(SEED_STREAM_NAMES, children)
    }
    return SeedManifest(root_seed=root_seed, streams=streams)


def select_learning_device(require_cuda: bool) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "run configuration requires CUDA, but CUDA is unavailable"
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_configs(
    system_config_path: str | Path,
    run_config_path: str | Path,
) -> tuple[ExperimentConfig, RunConfig, ExperimentAudit]:
    system = load_experiment_config(system_config_path)
    run = load_run_config(
        run_config_path,
        number_of_users=system.number_of_users,
        history_length=system.learning.history_length,
    )
    audit = audit_experiment(system)
    return system, run, audit


def _task_parameters(task: GeneratedTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "user_speeds_m_per_s": list(
            task.trajectory.user_speeds_m_per_s
        ),
        "user_rician_factor_db": list(
            task.trajectory.user_rician_factor_db
        ),
    }
