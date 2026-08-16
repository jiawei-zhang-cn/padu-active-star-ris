"""Independent training and evaluation workflow for PADU."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .audit import ExperimentAudit, finite_blocklength_rate_sinr_threshold
from .data import ComplexFeatureNormalizer
from .experiment import (
    _task_parameters,
    derive_seed_manifest,
    load_configs,
    select_learning_device,
)
from .padu_model import (
    PADUContext,
    PADUController,
    _channel_to_normalized_features,
)
from .simulation import create_fixed_deployment
from .task_generation import assert_disjoint_task_ids, generate_task_family
from .torch_physics import (
    surface_output_powers,
    torch_fbl_spectral_efficiency,
    torch_user_sinr,
)
from .training import set_reproducible_seed
from .units import db_to_linear


@dataclass(frozen=True)
class UnfoldingArchitectureSettings:
    unfolding_layers: int
    initializer_hidden_widths: tuple[int, ...]
    initial_primal_step_size: float
    initial_dual_step_size: float
    predictor_input_mode: str
    design_payload_mode: str
    maximum_design_payload_bits: float | None
    fixed_design_payload_bits: float | None
    optimization_update_mode: str = "standard"
    csi_representation_mode: str = "gru_point"
    statistical_scenario_count: int = 1
    statistical_scenario_seed: int | None = None
    probabilistic_uncertainty_conditioning: bool = False
    controller_refinement_mode: str = "primal_dual_unfolding"
    control_objective_mode: str = "sum_rate"


@dataclass(frozen=True)
class UnfoldingTrainingSettings:
    batch_size: int
    learning_rate: float
    epochs: int
    validation_interval: int
    weight_decay: float
    gradient_norm_limit: float
    data_loader_workers: int
    prediction_loss_weight: float
    qos_shortfall_penalty_weight: float
    joint_qos_shortfall_penalty_weight: float
    intermediate_layer_loss_weight: float
    joint_qos_loss_mode: str = "squared_max_sinr_shortfall"
    joint_qos_boundary_temperature: float | None = None
    training_strategy: str = "joint"
    predictor_pretraining_epochs: int = 1


@dataclass(frozen=True)
class UnfoldingEvaluationSettings:
    batch_size: int
    data_loader_workers: int


@dataclass(frozen=True)
class UnfoldingRunSettings:
    schema_version: int
    architecture: UnfoldingArchitectureSettings
    training: UnfoldingTrainingSettings
    evaluation: UnfoldingEvaluationSettings


class JointChannelWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Joint K-user windows with no trajectory-boundary leakage."""

    def __init__(
        self,
        trajectories: Sequence[np.ndarray],
        normalizer: ComplexFeatureNormalizer,
        history_length: int,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self._histories: list[Tensor] = []
        self._targets: list[Tensor] = []
        for trajectory in trajectories:
            value = np.asarray(trajectory, dtype=np.complex64)
            if value.ndim != 3:
                raise ValueError("trajectory must have shape (time, N, K)")
            if value.shape[0] <= history_length:
                raise ValueError("trajectory must be longer than history_length")
            per_user = value.transpose(0, 2, 1)
            normalized = normalizer.transform(per_user)
            for target_index in range(history_length, value.shape[0]):
                history = normalized[
                    target_index - history_length : target_index
                ].transpose(1, 0, 2)
                self._histories.append(torch.from_numpy(history.copy()))
                self._targets.append(torch.from_numpy(value[target_index].copy()))

    def __len__(self) -> int:
        return len(self._histories)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._histories[index], self._targets[index]


def load_unfolding_settings(path: str | Path) -> UnfoldingRunSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("unfolding configuration root must be an object")
    _exact_keys(raw, {"schema_version", "architecture", "training", "evaluation"}, "root")
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must equal 1")
    architecture = _object(raw["architecture"], "architecture")
    required_architecture_keys = {
        "unfolding_layers",
        "initializer_hidden_widths",
        "initial_primal_step_size",
        "initial_dual_step_size",
    }
    allowed_architecture_keys = {
        *required_architecture_keys,
        "predictor_input_mode",
        "design_payload_mode",
        "maximum_design_payload_bits",
        "fixed_design_payload_bits",
        "optimization_update_mode",
        "csi_representation_mode",
        "statistical_scenario_count",
        "statistical_scenario_seed",
        "probabilistic_uncertainty_conditioning",
        "controller_refinement_mode",
        "control_objective_mode",
    }
    missing_architecture_keys = sorted(
        required_architecture_keys - set(architecture)
    )
    unknown_architecture_keys = sorted(
        set(architecture) - allowed_architecture_keys
    )
    if missing_architecture_keys or unknown_architecture_keys:
        raise ValueError(
            "architecture keys are invalid; "
            f"missing={missing_architecture_keys}, "
            f"unknown={unknown_architecture_keys}"
        )
    training = _object(raw["training"], "training")
    required_training_keys = {
        "batch_size",
        "learning_rate",
        "epochs",
        "validation_interval",
        "weight_decay",
        "gradient_norm_limit",
        "data_loader_workers",
        "prediction_loss_weight",
        "qos_shortfall_penalty_weight",
        "joint_qos_shortfall_penalty_weight",
        "intermediate_layer_loss_weight",
    }
    allowed_training_keys = {
        *required_training_keys,
        "training_strategy",
        "predictor_pretraining_epochs",
        "joint_qos_loss_mode",
        "joint_qos_boundary_temperature",
    }
    missing_training_keys = sorted(required_training_keys - set(training))
    unknown_training_keys = sorted(set(training) - allowed_training_keys)
    if missing_training_keys or unknown_training_keys:
        raise ValueError(
            "training keys are invalid; "
            f"missing={missing_training_keys}, "
            f"unknown={unknown_training_keys}"
        )
    evaluation = _object(raw["evaluation"], "evaluation")
    _exact_keys(evaluation, {"batch_size", "data_loader_workers"}, "evaluation")
    widths = tuple(
        _positive_int(value, "architecture.initializer_hidden_widths")
        for value in architecture["initializer_hidden_widths"]
    )
    if not widths:
        raise ValueError("architecture.initializer_hidden_widths must be non-empty")
    design_payload_mode = _design_payload_mode(
        architecture.get("design_payload_mode", "business_minimum")
    )
    maximum_design_payload_bits = _optional_positive_float(
        architecture.get("maximum_design_payload_bits"),
        "architecture.maximum_design_payload_bits",
    )
    fixed_design_payload_bits = _optional_positive_float(
        architecture.get("fixed_design_payload_bits"),
        "architecture.fixed_design_payload_bits",
    )
    if (
        design_payload_mode in {
            "learned_userwise",
            "learned_userwise_causal_variation",
        }
        and maximum_design_payload_bits is None
    ):
        raise ValueError(
            "architecture.maximum_design_payload_bits is required for "
            "learned_userwise design payloads"
        )
    if (
        design_payload_mode == "business_minimum"
        and (
            maximum_design_payload_bits is not None
            or fixed_design_payload_bits is not None
        )
    ):
        raise ValueError(
            "architecture design payload bit fields must be omitted for "
            "business_minimum design payloads"
        )
    if (
        design_payload_mode
        in {"learned_userwise", "learned_userwise_causal_variation"}
        and fixed_design_payload_bits is not None
    ):
        raise ValueError(
            "architecture.fixed_design_payload_bits must be omitted for "
            "learned_userwise design payloads"
        )
    if design_payload_mode == "fixed_common":
        if fixed_design_payload_bits is None:
            raise ValueError(
                "architecture.fixed_design_payload_bits is required for "
                "fixed_common design payloads"
            )
        if maximum_design_payload_bits is not None:
            raise ValueError(
                "architecture.maximum_design_payload_bits must be omitted for "
                "fixed_common design payloads"
            )
    csi_representation_mode = _csi_representation_mode(
        architecture.get("csi_representation_mode", "gru_point")
    )
    controller_refinement_mode = _controller_refinement_mode(
        architecture.get(
            "controller_refinement_mode",
            "primal_dual_unfolding",
        )
    )
    probabilistic_uncertainty_conditioning = _boolean(
        architecture.get("probabilistic_uncertainty_conditioning", False),
        "architecture.probabilistic_uncertainty_conditioning",
    )
    statistical_scenario_count = _positive_int(
        architecture.get("statistical_scenario_count", 1),
        "architecture.statistical_scenario_count",
    )
    statistical_scenario_seed = architecture.get(
        "statistical_scenario_seed"
    )
    if csi_representation_mode == "gru_probabilistic":
        if statistical_scenario_count != 1:
            raise ValueError(
                "probabilistic GRU CSI requires one scenario"
            )
        if statistical_scenario_seed is not None:
            raise ValueError(
                "probabilistic GRU CSI requires no scenario seed"
            )
    elif csi_representation_mode == "latest_observed_csi":
        if (
            statistical_scenario_count != 1
            or statistical_scenario_seed is not None
        ):
            raise ValueError(
                "latest observed CSI requires one scenario and no scenario seed"
            )
    elif (
        statistical_scenario_count != 1
        or statistical_scenario_seed is not None
    ):
        raise ValueError(
            "GRU point CSI requires one scenario and no scenario seed"
        )
    control_objective_mode = _control_objective_mode(
        architecture.get("control_objective_mode", "sum_rate")
    )
    if (
        control_objective_mode == "max_min_normalized_payload"
        and design_payload_mode != "business_minimum"
    ):
        raise ValueError(
            "max-min normalized payload requires business_minimum "
            "design payloads"
        )
    joint_qos_loss_mode = _joint_qos_loss_mode(
        training.get(
            "joint_qos_loss_mode",
            "squared_max_sinr_shortfall",
        )
    )
    joint_qos_boundary_temperature = _optional_positive_float(
        training.get("joint_qos_boundary_temperature"),
        "training.joint_qos_boundary_temperature",
    )
    if (
        joint_qos_loss_mode == "smooth_payload_boundary"
        and joint_qos_boundary_temperature is None
    ):
        raise ValueError(
            "training.joint_qos_boundary_temperature is required for "
            "smooth_payload_boundary"
        )
    if (
        joint_qos_loss_mode == "squared_max_sinr_shortfall"
        and joint_qos_boundary_temperature is not None
    ):
        raise ValueError(
            "training.joint_qos_boundary_temperature must be omitted for "
            "squared_max_sinr_shortfall"
        )
    return UnfoldingRunSettings(
        schema_version=1,
        architecture=UnfoldingArchitectureSettings(
            unfolding_layers=_positive_int(
                architecture["unfolding_layers"], "architecture.unfolding_layers"
            ),
            initializer_hidden_widths=widths,
            initial_primal_step_size=_positive_float(
                architecture["initial_primal_step_size"],
                "architecture.initial_primal_step_size",
            ),
            initial_dual_step_size=_positive_float(
                architecture["initial_dual_step_size"],
                "architecture.initial_dual_step_size",
            ),
            predictor_input_mode=_predictor_input_mode(
                architecture.get("predictor_input_mode", "channel_only")
            ),
            design_payload_mode=design_payload_mode,
            maximum_design_payload_bits=maximum_design_payload_bits,
            fixed_design_payload_bits=fixed_design_payload_bits,
            optimization_update_mode=_optimization_update_mode(
                architecture.get("optimization_update_mode", "standard")
            ),
            csi_representation_mode=csi_representation_mode,
            statistical_scenario_count=statistical_scenario_count,
            statistical_scenario_seed=statistical_scenario_seed,
            probabilistic_uncertainty_conditioning=(
                probabilistic_uncertainty_conditioning
            ),
            controller_refinement_mode=controller_refinement_mode,
            control_objective_mode=control_objective_mode,
        ),
        training=UnfoldingTrainingSettings(
            batch_size=_positive_int(training["batch_size"], "training.batch_size"),
            learning_rate=_positive_float(
                training["learning_rate"], "training.learning_rate"
            ),
            epochs=_positive_int(training["epochs"], "training.epochs"),
            training_strategy=_training_strategy(
                training.get("training_strategy", "joint")
            ),
            predictor_pretraining_epochs=_positive_int(
                training.get(
                    "predictor_pretraining_epochs",
                    training["epochs"],
                ),
                "training.predictor_pretraining_epochs",
            ),
            validation_interval=_positive_int(
                training["validation_interval"],
                "training.validation_interval",
            ),
            weight_decay=_nonnegative_float(
                training["weight_decay"], "training.weight_decay"
            ),
            gradient_norm_limit=_positive_float(
                training["gradient_norm_limit"],
                "training.gradient_norm_limit",
            ),
            data_loader_workers=_nonnegative_int(
                training["data_loader_workers"],
                "training.data_loader_workers",
            ),
            prediction_loss_weight=_nonnegative_float(
                training["prediction_loss_weight"],
                "training.prediction_loss_weight",
            ),
            qos_shortfall_penalty_weight=_nonnegative_float(
                training["qos_shortfall_penalty_weight"],
                "training.qos_shortfall_penalty_weight",
            ),
            joint_qos_shortfall_penalty_weight=_nonnegative_float(
                training["joint_qos_shortfall_penalty_weight"],
                "training.joint_qos_shortfall_penalty_weight",
            ),
            intermediate_layer_loss_weight=_nonnegative_float(
                training["intermediate_layer_loss_weight"],
                "training.intermediate_layer_loss_weight",
            ),
            joint_qos_loss_mode=joint_qos_loss_mode,
            joint_qos_boundary_temperature=(
                joint_qos_boundary_temperature
            ),
        ),
        evaluation=UnfoldingEvaluationSettings(
            batch_size=_positive_int(
                evaluation["batch_size"], "evaluation.batch_size"
            ),
            data_loader_workers=_nonnegative_int(
                evaluation["data_loader_workers"],
                "evaluation.data_loader_workers",
            ),
        ),
    )


def train_padu(
    *,
    system_config_path: str | Path,
    run_config_path: str | Path,
    unfolding_config_path: str | Path,
    pretrained_predictor_directory: str | Path | None = None,
    output_root: str | Path,
) -> dict[str, Any]:
    system, run, audit = load_configs(system_config_path, run_config_path)
    settings = load_unfolding_settings(unfolding_config_path)
    if len(run.root_seeds) != 1:
        raise ValueError("PADU training requires one root seed")
    if (
        pretrained_predictor_directory is not None
        and settings.training.training_strategy != "predictor_then_controller"
    ):
        raise ValueError(
            "pretrained predictor reuse requires "
            "training_strategy=predictor_then_controller"
        )
    if (
        settings.architecture.csi_representation_mode
        == "gru_probabilistic"
        and settings.training.training_strategy
        != "predictor_then_controller"
    ):
        raise ValueError(
            "probabilistic GRU CSI requires "
            "training_strategy=predictor_then_controller"
        )
    if (
        settings.architecture.csi_representation_mode == "latest_observed_csi"
        and settings.training.training_strategy != "controller_only"
    ):
        raise ValueError(
            "latest observed CSI requires training_strategy=controller_only"
        )
    predictor_source = (
        None
        if pretrained_predictor_directory is None
        else Path(pretrained_predictor_directory)
    )
    if predictor_source is not None:
        _require_pretrained_predictor_files(predictor_source)
    device = select_learning_device(run.require_cuda)
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite output: {root}")
    root.mkdir(parents=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()

    root_seed = run.root_seeds[0]
    seed_manifest = derive_seed_manifest(root_seed)
    _write_json(root / "seeds.json", asdict(seed_manifest))
    deployment = create_fixed_deployment(
        system,
        np.random.default_rng(seed_manifest.streams["deployment"]),
    )
    families = _generate_selected_task_families(
        system=system,
        run=run,
        deployment=deployment,
        streams=seed_manifest.streams,
        family_names=(
            "gru_training",
            "gru_validation",
            "meta_training",
            "meta_validation",
        ),
    )
    assert_disjoint_task_ids(*families.values())
    task_parameters = {
        name: [_task_parameters(task) for task in tasks]
        for name, tasks in families.items()
    }
    _write_json(root / "task_parameters.json", task_parameters)
    training_trajectories = _task_channels(
        (*families["gru_training"], *families["meta_training"])
    )
    validation_trajectories = _task_channels(
        (*families["gru_validation"], *families["meta_validation"])
    )
    normalizer = ComplexFeatureNormalizer.fit(
        [value.transpose(0, 2, 1) for value in training_trajectories]
    )
    np.savez(
        root / "normalizer.npz",
        mean=normalizer.mean,
        standard_deviation=normalizer.standard_deviation,
    )
    training_dataset = JointChannelWindowDataset(
        training_trajectories,
        normalizer,
        system.learning.history_length,
    )
    validation_dataset = JointChannelWindowDataset(
        validation_trajectories,
        normalizer,
        system.learning.history_length,
    )
    set_reproducible_seed(seed_manifest.streams["joint_model_initialization"])
    model = _build_model(system, run, settings).to(device)
    model_dimensions = _model_dimensions(system, run, settings)
    pretrained_predictor = (
        None
        if predictor_source is None
        else _load_compatible_pretrained_predictor(
            source=predictor_source,
            seed_manifest=asdict(seed_manifest),
            task_parameters=task_parameters,
            normalizer=normalizer,
            model_dimensions=model_dimensions,
            device=device,
        )
    )
    context = _build_context(
        system,
        audit,
        run.physics_loss.feasible_mapping_epsilon,
        deployment.bs_to_surface_channel,
        device,
        settings,
    )
    mean = torch.from_numpy(normalizer.mean).to(device)
    standard_deviation = torch.from_numpy(normalizer.standard_deviation).to(device)
    training_loader = _preloaded_data_loader(
        training_dataset,
        settings.training.batch_size,
        True,
        device,
        seed_manifest.streams["joint_training"],
    )
    validation_loader = _preloaded_data_loader(
        validation_dataset,
        settings.training.batch_size,
        False,
        device,
        seed_manifest.streams["joint_training"],
    )
    start = perf_counter()
    if settings.training.training_strategy == "controller_only":
        training_result = _train_controller(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            context=context,
            mean=mean,
            standard_deviation=standard_deviation,
            settings=settings,
            final_checkpoint_path=checkpoints / "padu_controller.pt",
            model_dimensions=model_dimensions,
            training_stage="controller_only",
        )
        _write_csv(
            root / "controller_training_history.csv",
            training_result["controller_history"],
        )
        predictor_best_epoch = None
        predictor_best_validation = None
        predictor_selection_metric = None
        predictor_best_validation_nmse = None
        predictor_best_validation_nll = None
        controller_best_epoch = training_result["controller_best_epoch"]
        controller_best_validation = training_result[
            "controller_best_validation"
        ]
        best_epoch = controller_best_epoch
        best_validation = controller_best_validation
    elif settings.training.training_strategy == "joint":
        training_result = _train_joint_model(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            context=context,
            mean=mean,
            standard_deviation=standard_deviation,
            settings=settings,
            checkpoint_path=checkpoints / "padu_controller.pt",
            model_dimensions=model_dimensions,
        )
        _write_csv(root / "training_history.csv", training_result["history"])
        predictor_best_epoch = None
        predictor_best_validation = None
        predictor_selection_metric = None
        predictor_best_validation_nmse = None
        predictor_best_validation_nll = None
        controller_best_epoch = None
        controller_best_validation = None
        best_epoch = training_result["best_epoch"]
        best_validation = training_result["best_validation"]
    else:
        training_result = _train_predictor_then_controller(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            context=context,
            mean=mean,
            standard_deviation=standard_deviation,
            settings=settings,
            predictor_checkpoint_path=checkpoints / "predictor.pt",
            final_checkpoint_path=checkpoints / "padu_controller.pt",
            model_dimensions=model_dimensions,
            pretrained_predictor=pretrained_predictor,
        )
        if predictor_source is None:
            _write_csv(
                root / "predictor_pretraining_history.csv",
                training_result["predictor_history"],
            )
        else:
            shutil.copy2(
                predictor_source / "predictor_pretraining_history.csv",
                root / "predictor_pretraining_history.csv",
            )
            _write_json(
                root / "predictor_source.json",
                {
                    "pretrained_predictor_directory": str(
                        predictor_source
                    ),
                    "predictor_checkpoint": str(
                        predictor_source
                        / "checkpoints"
                        / "predictor.pt"
                    ),
                },
            )
        _write_csv(
            root / "controller_training_history.csv",
            training_result["controller_history"],
        )
        predictor_best_epoch = training_result["predictor_best_epoch"]
        predictor_best_validation = training_result[
            "predictor_best_validation"
        ]
        predictor_selection_metric = training_result[
            "predictor_selection_metric"
        ]
        predictor_best_validation_nmse = training_result[
            "predictor_best_validation_nmse"
        ]
        predictor_best_validation_nll = training_result[
            "predictor_best_validation_nll"
        ]
        controller_best_epoch = training_result["controller_best_epoch"]
        controller_best_validation = training_result[
            "controller_best_validation"
        ]
        best_epoch = controller_best_epoch
        best_validation = controller_best_validation
    training_time = perf_counter() - start
    summary = {
        "root_seed": root_seed,
        "device": str(device),
        "training_strategy": settings.training.training_strategy,
        "pretrained_predictor_reused": predictor_source is not None,
        "csi_representation_mode": (
            settings.architecture.csi_representation_mode
        ),
        "pretrained_predictor_directory": (
            None if predictor_source is None else str(predictor_source)
        ),
        "training_samples": len(training_dataset),
        "validation_samples": len(validation_dataset),
        "preloaded_training_tensors": True,
        "training_data_loader_workers_used": 0,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_validation,
        "predictor_best_epoch": predictor_best_epoch,
        "predictor_selection_metric": predictor_selection_metric,
        "predictor_best_validation_nmse": (
            predictor_best_validation_nmse
        ),
        "predictor_best_validation_nll": predictor_best_validation_nll,
        "controller_best_epoch": controller_best_epoch,
        "controller_best_validation_physical_loss": (
            controller_best_validation
        ),
        "training_time_s": training_time,
        "checkpoint": str(checkpoints / "padu_controller.pt"),
    }
    _write_json(root / "summary.json", summary)
    return summary


def evaluate_padu(
    *,
    system_config_path: str | Path,
    run_config_path: str | Path,
    unfolding_config_path: str | Path,
    checkpoint_seed_directory: str | Path,
    output_root: str | Path,
    domains: Sequence[str],
    perfect_next_slot_csi: bool = False,
) -> dict[str, Any]:
    system, run, audit = load_configs(system_config_path, run_config_path)
    settings = load_unfolding_settings(unfolding_config_path)
    selected_domains = tuple(domains)
    if not selected_domains or len(set(selected_domains)) != len(selected_domains):
        raise ValueError("domains must be non-empty and unique")
    if set(selected_domains) - {"in_domain", "out_of_domain"}:
        raise ValueError("domains contain an unknown value")
    if len(run.root_seeds) != 1:
        raise ValueError("PADU evaluation requires one root seed")
    checkpoint_root = Path(checkpoint_seed_directory)
    checkpoint_seeds = json.loads(
        (checkpoint_root / "seeds.json").read_text(encoding="utf-8")
    )
    training_seed = int(checkpoint_seeds["root_seed"])
    training_manifest = derive_seed_manifest(training_seed)
    device = select_learning_device(run.require_cuda)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    deployment = create_fixed_deployment(
        system,
        np.random.default_rng(training_manifest.streams["deployment"]),
    )
    evaluation_manifest = derive_seed_manifest(run.root_seeds[0])
    requested_families = tuple(
        "in_domain_test" if domain == "in_domain" else "out_of_domain_test"
        for domain in selected_domains
    )
    families = _generate_selected_task_families(
        system=system,
        run=run,
        deployment=deployment,
        streams=evaluation_manifest.streams,
        family_names=requested_families,
    )
    assert_disjoint_task_ids(*families.values())
    with np.load(checkpoint_root / "normalizer.npz") as archive:
        normalizer = ComplexFeatureNormalizer(
            mean=np.asarray(archive["mean"], dtype=np.float32),
            standard_deviation=np.asarray(
                archive["standard_deviation"], dtype=np.float32
            ),
        )
    model = _build_model(system, run, settings).to(device)
    saved = torch.load(
        checkpoint_root / "checkpoints" / "padu_controller.pt",
        map_location=device,
        weights_only=True,
    )
    if saved["model_dimensions"] != _model_dimensions(system, run, settings):
        raise ValueError("checkpoint model dimensions do not match configuration")
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    context = _build_context(
        system,
        audit,
        run.physics_loss.feasible_mapping_epsilon,
        deployment.bs_to_surface_channel,
        device,
        settings,
    )
    mean = torch.from_numpy(normalizer.mean).to(device)
    standard_deviation = torch.from_numpy(normalizer.standard_deviation).to(device)
    rows: list[dict[str, Any]] = []
    domain_map = {
        "in_domain": "in_domain_test",
        "out_of_domain": "out_of_domain_test",
    }
    for domain in selected_domains:
        for task in families[domain_map[domain]]:
            dataset = JointChannelWindowDataset(
                [task.trajectory.surface_to_user_channels],
                normalizer,
                system.learning.history_length,
            )
            loader = _data_loader(
                dataset,
                settings.evaluation.batch_size,
                False,
                settings.evaluation.data_loader_workers,
                device,
                evaluation_manifest.streams[f"{domain}_execution"],
            )
            target_slot = system.learning.history_length
            for batch in loader:
                histories, true_channels, model_arguments = (
                    _unpack_model_batch(batch)
                )
                histories = histories.to(device, non_blocking=True)
                true_channels = true_channels.to(device, non_blocking=True)
                model_arguments = {
                    name: value.to(device, non_blocking=True)
                    for name, value in model_arguments.items()
                }
                _synchronize(device)
                start = perf_counter()
                output_value = model(
                    histories,
                    mean,
                    standard_deviation,
                    context,
                    perfect_next_slot_channel=(
                        true_channels if perfect_next_slot_csi else None
                    ),
                    **model_arguments,
                )
                _synchronize(device)
                elapsed = perf_counter() - start
                with torch.no_grad():
                    sinr = torch_user_sinr(
                        bs_to_surface_channel=context.bs_to_surface_channel,
                        surface_to_user_channel=true_channels,
                        controls=output_value.controls,
                        reflection_user_mask=context.reflection_user_mask,
                        surface_noise_power_watt=context.surface_noise_power_watt,
                        receiver_noise_power_watt=context.receiver_noise_power_watt,
                    )
                    rate = torch_fbl_spectral_efficiency(
                        sinr,
                        blocklength=context.blocklength,
                        decoding_error_probability=context.decoding_error_probability,
                    )
                    throughput = (
                        (1.0 - context.decoding_error_probability)
                        * system.noise.bandwidth_hz
                        * torch.sum(rate, dim=-1)
                    )
                    satisfied = sinr >= context.minimum_fbl_sinr
                    nmse = torch.sum(
                        torch.abs(output_value.predicted_channel - true_channels) ** 2,
                        dim=(-2, -1),
                    ) / torch.clamp(
                        torch.sum(torch.abs(true_channels) ** 2, dim=(-2, -1)),
                        min=1.0e-30,
                    )
                    violation = _maximum_hardware_violation(
                        output_value.controls, context
                    )
                batch_size = histories.shape[0]
                for index in range(batch_size):
                    rows.append(
                        {
                            "root_seed": run.root_seeds[0],
                            "domain": domain,
                            "scheme": _evaluation_scheme(
                                perfect_next_slot_csi=perfect_next_slot_csi,
                                settings=settings,
                            ),
                            "trajectory_id": task.task_id,
                            "target_slot": target_slot + index,
                            "prediction_nmse": float(nmse[index].cpu()),
                            "actual_weighted_throughput_bps": float(
                                throughput[index].cpu()
                            ),
                            "joint_qos_outage": int(
                                not bool(torch.all(satisfied[index]).cpu())
                            ),
                            "minimum_actual_sinr": float(
                                torch.min(sinr[index]).cpu()
                            ),
                            "maximum_constraint_violation": float(
                                violation[index].cpu()
                            ),
                            **_design_payload_row(
                                output_value.design_payload_bits[index]
                            ),
                            **_causal_variation_row(
                                output_value.causal_channel_variation[index]
                            ),
                            **_innovation_scale_row(
                                output_value.
                                conditional_innovation_standard_deviation,
                                index,
                            ),
                            "control_total_s": elapsed / batch_size,
                        }
                    )
                target_slot += batch_size
    _write_csv(output / "slot_results.csv", rows)
    summary = _summarize_evaluation(rows, output)
    summary["csi_source"] = (
        "perfect_next_slot_csi"
        if perfect_next_slot_csi
        else (
            {
                "gru_probabilistic": (
                    "probabilistic_gru_next_slot_distribution"
                ),
                "gru_point": "gru_next_slot_prediction",
                "latest_observed_csi": "latest_observed_csi",
            }[settings.architecture.csi_representation_mode]
        )
    )
    _write_json(output / "summary.json", summary)
    _write_json(output / "seeds.json", asdict(evaluation_manifest))
    _write_json(
        output / "task_parameters.json",
        {
            name: [_task_parameters(task) for task in tasks]
            for name, tasks in families.items()
        },
    )
    return summary


def _evaluation_scheme(
    *,
    perfect_next_slot_csi: bool,
    settings: UnfoldingRunSettings,
) -> str:
    if perfect_next_slot_csi:
        return "perfect_next_slot_csi_primal_dual_unfolding"
    if settings.architecture.csi_representation_mode == "gru_point":
        return "padu_primal_dual_unfolding"
    if settings.architecture.csi_representation_mode == "latest_observed_csi":
        return "without_csi_prediction_primal_dual_unfolding"
    if settings.architecture.csi_representation_mode == "gru_probabilistic":
        if (
            settings.architecture.control_objective_mode
            == "max_min_normalized_payload"
        ):
            if settings.architecture.statistical_scenario_count == 1:
                if (
                    settings.architecture.controller_refinement_mode
                    == "initializer_only"
                ):
                    return (
                        "probabilistic_gru_mean_scenario_without_"
                        "unfolding_refinement"
                    )
                if not (
                    settings.architecture
                    .probabilistic_uncertainty_conditioning
                ):
                    return (
                        "probabilistic_gru_mean_scenario_without_uncertainty_"
                        "conditioning_max_min_payload_primal_dual_unfolding"
                    )
                return (
                    "probabilistic_gru_mean_scenario_max_min_payload_"
                    "primal_dual_unfolding"
                )
        raise ValueError("probabilistic GRU PADU requires max-min objective")
    raise ValueError("unsupported PADU evaluation scheme")


def _train_joint_model(
    *,
    model: PADUController,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    context: PADUContext,
    mean: Tensor,
    standard_deviation: Tensor,
    settings: UnfoldingRunSettings,
    checkpoint_path: Path,
    model_dimensions: dict[str, Any],
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.training.epochs + 1):
        training_metrics = _run_epoch(
            model=model,
            loader=training_loader,
            context=context,
            mean=mean,
            standard_deviation=standard_deviation,
            settings=settings.training,
            optimizer=optimizer,
        )
        should_validate = _should_validate(epoch, settings.training)
        validation_metrics = (
            _run_epoch(
                model=model,
                loader=validation_loader,
                context=context,
                mean=mean,
                standard_deviation=standard_deviation,
                settings=settings.training,
                optimizer=None,
            )
            if should_validate
            else None
        )
        history.append(
            _joint_history_row(
                epoch,
                training_metrics,
                validation_metrics,
                should_validate,
            )
        )
        _print_training_progress(
            "joint",
            epoch,
            settings.training.epochs,
            training_metrics["total_loss"],
            None if validation_metrics is None else validation_metrics["total_loss"],
        )
        if (
            validation_metrics is not None
            and validation_metrics["total_loss"] < best_validation
        ):
            best_validation = validation_metrics["total_loss"]
            best_epoch = epoch
            _save_unfolding_checkpoint(
                path=checkpoint_path,
                model=model,
                epoch=epoch,
                validation_total_loss=best_validation,
                model_dimensions=model_dimensions,
                settings=settings,
                training_stage="joint",
            )
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
    }


def _train_predictor_then_controller(
    *,
    model: PADUController,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    context: PADUContext,
    mean: Tensor,
    standard_deviation: Tensor,
    settings: UnfoldingRunSettings,
    predictor_checkpoint_path: Path,
    final_checkpoint_path: Path,
    model_dimensions: dict[str, Any],
    pretrained_predictor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predictor_history: list[dict[str, Any]] = []
    probabilistic = (
        model.csi_representation_mode == "gru_probabilistic"
    )
    selection_metric = "nll" if probabilistic else "nmse"
    if pretrained_predictor is None:
        predictor_optimizer = torch.optim.AdamW(
            model.predictor.parameters(),
            lr=settings.training.learning_rate,
            weight_decay=settings.training.weight_decay,
        )
        predictor_best_validation = float("inf")
        predictor_best_epoch = 0
        for epoch in range(1, settings.training.predictor_pretraining_epochs + 1):
            training_metrics = (
                _run_probabilistic_predictor_epoch(
                    model=model,
                    loader=training_loader,
                    mean=mean,
                    standard_deviation=standard_deviation,
                    gradient_norm_limit=(
                        settings.training.gradient_norm_limit
                    ),
                    optimizer=predictor_optimizer,
                )
                if probabilistic
                else {
                    "nmse": _run_predictor_epoch(
                        model=model,
                        loader=training_loader,
                        mean=mean,
                        standard_deviation=standard_deviation,
                        gradient_norm_limit=(
                            settings.training.gradient_norm_limit
                        ),
                        optimizer=predictor_optimizer,
                    )
                }
            )
            should_validate = (
                epoch % settings.training.validation_interval == 0
                or epoch == settings.training.predictor_pretraining_epochs
            )
            validation_metrics = (
                (
                    _run_probabilistic_predictor_epoch(
                        model=model,
                        loader=validation_loader,
                        mean=mean,
                        standard_deviation=standard_deviation,
                        gradient_norm_limit=(
                            settings.training.gradient_norm_limit
                        ),
                        optimizer=None,
                    )
                    if probabilistic
                    else {
                        "nmse": _run_predictor_epoch(
                            model=model,
                            loader=validation_loader,
                            mean=mean,
                            standard_deviation=standard_deviation,
                            gradient_norm_limit=(
                                settings.training.gradient_norm_limit
                            ),
                            optimizer=None,
                        )
                    }
                )
                if should_validate
                else None
            )
            history_row = {
                "epoch": epoch,
                "training_prediction_nmse": training_metrics["nmse"],
                "validation_performed": should_validate,
                "validation_prediction_nmse": (
                    None
                    if validation_metrics is None
                    else validation_metrics["nmse"]
                ),
            }
            if probabilistic:
                history_row.update(
                    {
                        "training_prediction_nll": (
                            training_metrics["nll"]
                        ),
                        "validation_prediction_nll": (
                            None
                            if validation_metrics is None
                            else validation_metrics["nll"]
                        ),
                    }
                )
            predictor_history.append(history_row)
            training_selection = training_metrics[selection_metric]
            validation_selection = (
                None
                if validation_metrics is None
                else validation_metrics[selection_metric]
            )
            _print_training_progress(
                "predictor",
                epoch,
                settings.training.predictor_pretraining_epochs,
                training_selection,
                validation_selection,
            )
            if (
                validation_selection is not None
                and validation_selection < predictor_best_validation
            ):
                predictor_best_validation = validation_selection
                predictor_best_epoch = epoch
                checkpoint = {
                    "predictor_state_dict": model.predictor.state_dict(),
                    "epoch": epoch,
                    "validation_prediction_nmse": (
                        validation_metrics["nmse"]
                    ),
                    "predictor_selection_metric": selection_metric,
                    "model_dimensions": model_dimensions,
                    "unfolding_settings": asdict(settings),
                }
                if probabilistic:
                    checkpoint["validation_prediction_nll"] = (
                        validation_metrics["nll"]
                    )
                torch.save(checkpoint, predictor_checkpoint_path)
        predictor_saved = torch.load(
            predictor_checkpoint_path,
            map_location=mean.device,
            weights_only=True,
        )
    else:
        predictor_saved = pretrained_predictor
        predictor_best_epoch = int(predictor_saved["epoch"])
        predictor_best_validation = float(
            predictor_saved[
                "validation_prediction_nll"
                if probabilistic
                else "validation_prediction_nmse"
            ]
        )
        _advance_loader_states_past_predictor_pretraining(
            training_loader=training_loader,
            validation_loader=validation_loader,
            predictor_checkpoint=predictor_saved,
        )
        torch.save(predictor_saved, predictor_checkpoint_path)
        print(
            "[PADU] stage=predictor reused "
            f"epoch={predictor_best_epoch} "
            f"validation={predictor_best_validation:.6g}",
            flush=True,
        )
    model.predictor.load_state_dict(predictor_saved["predictor_state_dict"])
    for parameter in model.predictor.parameters():
        parameter.requires_grad_(False)

    controller_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not controller_parameters:
        raise RuntimeError("controller training has no trainable parameters")
    controller_optimizer = torch.optim.AdamW(
        controller_parameters,
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    controller_history: list[dict[str, Any]] = []
    controller_best_validation = float("inf")
    controller_best_epoch = 0
    for epoch in range(1, settings.training.epochs + 1):
        training_metrics = _run_epoch(
            model=model,
            loader=training_loader,
            context=context,
            mean=mean,
            standard_deviation=standard_deviation,
            settings=settings.training,
            optimizer=controller_optimizer,
            loss_mode="physical_only",
        )
        should_validate = _should_validate(epoch, settings.training)
        validation_metrics = (
            _run_epoch(
                model=model,
                loader=validation_loader,
                context=context,
                mean=mean,
                standard_deviation=standard_deviation,
                settings=settings.training,
                optimizer=None,
                loss_mode="physical_only",
            )
            if should_validate
            else None
        )
        controller_history.append(
            _joint_history_row(
                epoch,
                training_metrics,
                validation_metrics,
                should_validate,
            )
        )
        _print_training_progress(
            "controller",
            epoch,
            settings.training.epochs,
            training_metrics["physical_loss"],
            (
                None
                if validation_metrics is None
                else validation_metrics["physical_loss"]
            ),
        )
        if (
            validation_metrics is not None
            and validation_metrics["physical_loss"] < controller_best_validation
        ):
            controller_best_validation = validation_metrics["physical_loss"]
            controller_best_epoch = epoch
            _save_unfolding_checkpoint(
                path=final_checkpoint_path,
                model=model,
                epoch=epoch,
                validation_total_loss=controller_best_validation,
                model_dimensions=model_dimensions,
                settings=settings,
                training_stage="frozen_predictor_controller",
                extra={
                    "predictor_best_epoch": predictor_best_epoch,
                    "predictor_selection_metric": selection_metric,
                    "predictor_best_validation_nmse": float(
                        predictor_saved["validation_prediction_nmse"]
                    ),
                    **(
                        {
                            "predictor_best_validation_nll": float(
                                predictor_saved[
                                    "validation_prediction_nll"
                                ]
                            )
                        }
                        if probabilistic
                        else {}
                    ),
                },
            )
    return {
        "predictor_history": predictor_history,
        "controller_history": controller_history,
        "predictor_best_epoch": predictor_best_epoch,
        "predictor_best_validation": predictor_best_validation,
        "predictor_selection_metric": selection_metric,
        "predictor_best_validation_nmse": float(
            predictor_saved["validation_prediction_nmse"]
        ),
        "predictor_best_validation_nll": (
            float(predictor_saved["validation_prediction_nll"])
            if probabilistic
            else None
        ),
        "controller_best_epoch": controller_best_epoch,
        "controller_best_validation": controller_best_validation,
    }


def _train_controller(
    *,
    model: PADUController,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    context: PADUContext,
    mean: Tensor,
    standard_deviation: Tensor,
    settings: UnfoldingRunSettings,
    final_checkpoint_path: Path,
    model_dimensions: dict[str, Any],
    training_stage: str = "controller",
) -> dict[str, Any]:
    if model.predictor is not None:
        for parameter in model.predictor.parameters():
            parameter.requires_grad_(False)
    controller_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        controller_parameters,
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.training.epochs + 1):
        training_metrics = _run_epoch(
            model=model,
            loader=training_loader,
            context=context,
            mean=mean,
            standard_deviation=standard_deviation,
            settings=settings.training,
            optimizer=optimizer,
            loss_mode="physical_only",
        )
        should_validate = _should_validate(epoch, settings.training)
        validation_metrics = (
            _run_epoch(
                model=model,
                loader=validation_loader,
                context=context,
                mean=mean,
                standard_deviation=standard_deviation,
                settings=settings.training,
                optimizer=None,
                loss_mode="physical_only",
            )
            if should_validate
            else None
        )
        history.append(
            _joint_history_row(
                epoch,
                training_metrics,
                validation_metrics,
                should_validate,
            )
        )
        _print_training_progress(
            "controller",
            epoch,
            settings.training.epochs,
            training_metrics["physical_loss"],
            (
                None
                if validation_metrics is None
                else validation_metrics["physical_loss"]
            ),
        )
        if (
            validation_metrics is not None
            and validation_metrics["physical_loss"] < best_validation
        ):
            best_validation = validation_metrics["physical_loss"]
            best_epoch = epoch
            _save_unfolding_checkpoint(
                path=final_checkpoint_path,
                model=model,
                epoch=epoch,
                validation_total_loss=best_validation,
                model_dimensions=model_dimensions,
                settings=settings,
                training_stage=training_stage,
            )
    return {
        "controller_history": history,
        "controller_best_epoch": best_epoch,
        "controller_best_validation": best_validation,
    }


def _advance_loader_states_past_predictor_pretraining(
    *,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    predictor_checkpoint: dict[str, Any],
) -> None:
    saved_settings = predictor_checkpoint.get("unfolding_settings")
    if not isinstance(saved_settings, dict):
        raise ValueError(
            "pretrained predictor checkpoint has no unfolding_settings"
        )
    saved_training = saved_settings.get("training")
    if not isinstance(saved_training, dict):
        raise ValueError(
            "pretrained predictor checkpoint has no training settings"
        )
    epochs = _positive_int(
        saved_training.get("predictor_pretraining_epochs"),
        "pretrained predictor training.predictor_pretraining_epochs",
    )
    validation_interval = _positive_int(
        saved_training.get("validation_interval"),
        "pretrained predictor training.validation_interval",
    )
    for epoch in range(1, epochs + 1):
        for _ in training_loader:
            pass
        if epoch % validation_interval == 0 or epoch == epochs:
            for _ in validation_loader:
                pass


def _require_pretrained_predictor_files(source: Path) -> None:
    required = (
        source / "seeds.json",
        source / "task_parameters.json",
        source / "normalizer.npz",
        source / "predictor_pretraining_history.csv",
        source / "checkpoints" / "predictor.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "pretrained predictor directory is incomplete; "
            f"missing={missing}"
        )


def _load_compatible_pretrained_predictor(
    *,
    source: Path,
    seed_manifest: dict[str, Any],
    task_parameters: dict[str, Any],
    normalizer: ComplexFeatureNormalizer,
    model_dimensions: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    source_seeds = json.loads(
        (source / "seeds.json").read_text(encoding="utf-8")
    )
    if source_seeds != seed_manifest:
        raise ValueError("pretrained predictor seed manifest does not match")
    source_tasks = json.loads(
        (source / "task_parameters.json").read_text(encoding="utf-8")
    )
    if source_tasks != task_parameters:
        raise ValueError("pretrained predictor task parameters do not match")
    with np.load(source / "normalizer.npz") as archive:
        source_mean = np.asarray(archive["mean"], dtype=np.float32)
        source_standard_deviation = np.asarray(
            archive["standard_deviation"], dtype=np.float32
        )
    if not np.array_equal(source_mean, normalizer.mean):
        raise ValueError("pretrained predictor normalizer mean does not match")
    if not np.array_equal(
        source_standard_deviation,
        normalizer.standard_deviation,
    ):
        raise ValueError(
            "pretrained predictor normalizer standard deviation does not match"
        )
    saved = torch.load(
        source / "checkpoints" / "predictor.pt",
        map_location=device,
        weights_only=True,
    )
    if _predictor_dimensions(saved["model_dimensions"]) != _predictor_dimensions(
        model_dimensions
    ):
        raise ValueError("pretrained predictor model dimensions do not match")
    return saved


def _predictor_dimensions(model_dimensions: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "number_of_surface_elements",
        "number_of_users",
        "history_length",
        "gru_hidden_size",
        "gru_number_of_layers",
    )
    dimensions = {key: model_dimensions[key] for key in keys}
    dimensions["predictor_input_mode"] = model_dimensions.get(
        "predictor_input_mode",
        "channel_only",
    )
    dimensions["csi_representation_mode"] = model_dimensions.get(
        "csi_representation_mode",
        "gru_point",
    )
    return dimensions


def _run_epoch(
    *,
    model: PADUController,
    loader: DataLoader,
    context: PADUContext,
    mean: Tensor,
    standard_deviation: Tensor,
    settings: UnfoldingTrainingSettings,
    optimizer: torch.optim.Optimizer | None,
    loss_mode: str = "joint",
) -> dict[str, float]:
    if loss_mode not in {"joint", "physical_only"}:
        raise ValueError("loss_mode is unsupported")
    training = optimizer is not None
    model.train(training)
    if model.predictor is not None and not any(
        parameter.requires_grad for parameter in model.predictor.parameters()
    ):
        model.predictor.eval()
    totals = {"total_loss": 0.0, "prediction_loss": 0.0, "physical_loss": 0.0}
    samples = 0
    for batch in loader:
        histories, true_channels, model_arguments = _unpack_model_batch(batch)
        histories = histories.to(mean.device, non_blocking=True)
        true_channels = true_channels.to(mean.device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = model(
            histories,
            mean,
            standard_deviation,
            context,
            **{
                name: value.to(mean.device, non_blocking=True)
                for name, value in model_arguments.items()
            },
        )
        joint_loss, components = _joint_loss(
            output.predicted_channel,
            output.layer_controls,
            true_channels,
            context,
            settings,
            control_objective_mode=model.control_objective_mode,
            scenario_sinr=(
                output.layer_scenario_sinr
                if model.control_objective_mode
                == "max_min_normalized_payload"
                else None
            ),
        )
        loss = (
            joint_loss
            if loss_mode == "joint"
            else components["physical_loss"]
        )
        if optimizer is not None:
            loss.backward()
            clip_grad_norm_(model.parameters(), settings.gradient_norm_limit)
            optimizer.step()
        batch_size = histories.shape[0]
        samples += batch_size
        totals["total_loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] += float(value.detach()) * batch_size
    return {name: value / samples for name, value in totals.items()}


def _unpack_model_batch(
    batch: Sequence[Tensor],
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if len(batch) == 2:
        return batch[0], batch[1], {}
    raise ValueError("model batch must contain two tensors")


def _run_predictor_epoch(
    *,
    model: PADUController,
    loader: DataLoader,
    mean: Tensor,
    standard_deviation: Tensor,
    gradient_norm_limit: float,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.predictor.train(training)
    total = 0.0
    samples = 0
    for histories, true_channels in loader:
        histories = histories.to(mean.device, non_blocking=True)
        true_channels = true_channels.to(mean.device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        predicted_channel, _, _ = model.predict_channel(
            histories,
            mean,
            standard_deviation,
        )
        loss = _prediction_nmse(predicted_channel, true_channels)
        if optimizer is not None:
            loss.backward()
            clip_grad_norm_(model.predictor.parameters(), gradient_norm_limit)
            optimizer.step()
        batch_size = histories.shape[0]
        total += float(loss.detach()) * batch_size
        samples += batch_size
    return total / samples


def _run_probabilistic_predictor_epoch(
    *,
    model: PADUController,
    loader: DataLoader,
    mean: Tensor,
    standard_deviation: Tensor,
    gradient_norm_limit: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.predictor.train(training)
    total_nll = 0.0
    total_nmse = 0.0
    samples = 0
    for histories, true_channels in loader:
        histories = histories.to(mean.device, non_blocking=True)
        true_channels = true_channels.to(mean.device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        (
            predicted_channel,
            normalized_prediction,
            _,
            normalized_standard_deviation,
        ) = model.predict_channel_distribution(
            histories,
            mean,
            standard_deviation,
        )
        normalized_target = _channel_to_normalized_features(
            true_channels,
            mean,
            standard_deviation,
        )
        nll = _isotropic_normalized_feature_nll(
            normalized_prediction,
            normalized_standard_deviation,
            normalized_target,
        )
        nmse = _prediction_nmse(predicted_channel, true_channels)
        if optimizer is not None:
            nll.backward()
            clip_grad_norm_(
                model.predictor.parameters(), gradient_norm_limit
            )
            optimizer.step()
        batch_size = histories.shape[0]
        total_nll += float(nll.detach()) * batch_size
        total_nmse += float(nmse.detach()) * batch_size
        samples += batch_size
    return {
        "nll": total_nll / samples,
        "nmse": total_nmse / samples,
    }


def _isotropic_normalized_feature_nll(
    normalized_prediction: Tensor,
    normalized_standard_deviation: Tensor,
    normalized_target: Tensor,
) -> Tensor:
    """Gaussian NLL for one shared real-feature scale per user."""
    if normalized_prediction.shape != normalized_target.shape:
        raise ValueError(
            "normalized prediction and target shapes must match"
        )
    if normalized_prediction.ndim != 3:
        raise ValueError(
            "normalized prediction must have shape (batch, K, 2N)"
        )
    if normalized_standard_deviation.shape != (
        normalized_prediction.shape[0],
        normalized_prediction.shape[1],
    ):
        raise ValueError(
            "normalized standard deviation must have shape (batch, K)"
        )
    if torch.any(normalized_standard_deviation <= 0.0):
        raise ValueError("normalized standard deviation must be positive")
    standardized_error = (
        normalized_target - normalized_prediction
    ) / normalized_standard_deviation.unsqueeze(-1)
    per_feature_nll = (
        0.5 * standardized_error.square()
        + torch.log(normalized_standard_deviation).unsqueeze(-1)
    )
    return torch.mean(per_feature_nll)


def _prediction_nmse(predicted_channel: Tensor, true_channel: Tensor) -> Tensor:
    normalized_error = torch.sum(
        torch.abs(predicted_channel - true_channel) ** 2, dim=(-2, -1)
    ) / torch.clamp(
        torch.sum(torch.abs(true_channel) ** 2, dim=(-2, -1)),
        min=1.0e-30,
    )
    return torch.mean(normalized_error)


def _should_validate(
    epoch: int,
    settings: UnfoldingTrainingSettings,
) -> bool:
    return (
        epoch % settings.validation_interval == 0
        or epoch == settings.epochs
    )


def _joint_history_row(
    epoch: int,
    training_metrics: dict[str, float],
    validation_metrics: dict[str, float] | None,
    validation_performed: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        **{
            f"training_{key}": value
            for key, value in training_metrics.items()
        },
        "validation_performed": validation_performed,
        **(
            {
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            }
            if validation_metrics is not None
            else {
                "validation_total_loss": None,
                "validation_prediction_loss": None,
                "validation_physical_loss": None,
            }
        ),
    }


def _print_training_progress(
    stage: str,
    epoch: int,
    epochs: int,
    training_value: float,
    validation_value: float | None,
) -> None:
    validation_text = (
        f" validation={validation_value:.6g}"
        if validation_value is not None
        else " validation=skipped"
    )
    print(
        "[PADU] "
        f"stage={stage} epoch={epoch}/{epochs} "
        f"train={training_value:.6g}"
        + validation_text,
        flush=True,
    )


def _save_unfolding_checkpoint(
    *,
    path: Path,
    model: PADUController,
    epoch: int,
    validation_total_loss: float,
    model_dimensions: dict[str, Any],
    settings: UnfoldingRunSettings,
    training_stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "validation_total_loss": validation_total_loss,
        "model_dimensions": model_dimensions,
        "unfolding_settings": asdict(settings),
        "training_stage": training_stage,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def _joint_loss(
    predicted_channel: Tensor,
    layer_controls: Sequence,
    true_channel: Tensor,
    context: PADUContext,
    settings: UnfoldingTrainingSettings,
    control_objective_mode: str = "sum_rate",
    scenario_sinr: Sequence[Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if scenario_sinr is not None and len(scenario_sinr) != len(layer_controls):
        raise ValueError("scenario SINR count must match layer control count")
    prediction_loss = _prediction_nmse(predicted_channel, true_channel)
    layer_losses = []
    for layer_index, controls in enumerate(layer_controls):
        if scenario_sinr is None:
            sinr = torch_user_sinr(
                bs_to_surface_channel=context.bs_to_surface_channel,
                surface_to_user_channel=true_channel,
                controls=controls,
                reflection_user_mask=context.reflection_user_mask,
                surface_noise_power_watt=context.surface_noise_power_watt,
                receiver_noise_power_watt=context.receiver_noise_power_watt,
            )
        else:
            sinr = scenario_sinr[layer_index]
        rate = torch_fbl_spectral_efficiency(
            sinr,
            blocklength=context.blocklength,
            decoding_error_probability=context.decoding_error_probability,
        )
        positive = context.minimum_fbl_sinr > 0.0
        denominator = torch.where(
            positive, context.minimum_fbl_sinr, torch.ones_like(context.minimum_fbl_sinr)
        )
        shortfall = torch.where(
            positive,
            torch.relu((context.minimum_fbl_sinr - sinr) / denominator),
            torch.zeros_like(sinr),
        )
        if settings.joint_qos_loss_mode == "squared_max_sinr_shortfall":
            if control_objective_mode == "max_min_normalized_payload":
                joint_penalty = torch.mean(
                    torch.amax(
                        shortfall.square(),
                        dim=tuple(range(1, shortfall.ndim)),
                    )
                )
            else:
                joint_penalty = torch.mean(
                    torch.amax(shortfall.square(), dim=-1)
                )
        else:
            temperature = settings.joint_qos_boundary_temperature
            if temperature is None:
                raise RuntimeError(
                    "joint QoS boundary temperature is missing"
                )
            minimum_payload = context.minimum_design_payload_bits.to(
                dtype=rate.dtype,
                device=rate.device,
            )
            positive_payload = minimum_payload > 0.0
            payload_denominator = torch.where(
                positive_payload,
                minimum_payload,
                torch.ones_like(minimum_payload),
            )
            normalized_payload_gap = torch.where(
                positive_payload,
                (
                    minimum_payload.unsqueeze(0)
                    - context.blocklength * rate
                )
                / payload_denominator.unsqueeze(0),
                torch.zeros_like(rate),
            )
            zero_reference = torch.zeros(
                *normalized_payload_gap.shape[:-1],
                1,
                dtype=rate.dtype,
                device=rate.device,
            )
            joint_penalty = temperature * torch.mean(
                torch.logsumexp(
                    torch.cat(
                        (
                            zero_reference,
                            normalized_payload_gap / temperature,
                        ),
                        dim=-1,
                    ),
                    dim=-1,
                )
            )
        if control_objective_mode == "sum_rate":
            utility_loss = -torch.mean(torch.sum(rate, dim=-1))
        elif control_objective_mode == "max_min_normalized_payload":
            minimum_payload = context.minimum_design_payload_bits.to(
                dtype=rate.dtype,
                device=rate.device,
            )
            if torch.any(minimum_payload <= 0.0):
                raise ValueError(
                    "max-min normalized payload requires positive "
                    "business payloads"
                )
            normalized_payload = context.blocklength * rate / minimum_payload.unsqueeze(-2)
            utility_loss = -torch.mean(
                torch.amin(normalized_payload, dim=tuple(range(1, normalized_payload.ndim)))
            )
        else:
            raise ValueError("control_objective_mode is unsupported")
        layer_losses.append(
            utility_loss
            + settings.qos_shortfall_penalty_weight
            * (
                torch.mean(
                    torch.sum(shortfall.square(), dim=(-2, -1))
                )
                if control_objective_mode
                == "max_min_normalized_payload"
                else torch.mean(torch.sum(shortfall.square(), dim=-1))
            )
            + settings.joint_qos_shortfall_penalty_weight
            * joint_penalty
        )
    physical_loss = layer_losses[-1]
    if len(layer_losses) > 1:
        physical_loss = physical_loss + settings.intermediate_layer_loss_weight * torch.mean(
            torch.stack(layer_losses[:-1])
        )
    total = settings.prediction_loss_weight * prediction_loss + physical_loss
    return total, {
        "prediction_loss": prediction_loss,
        "physical_loss": physical_loss,
    }


def _build_model(system, run, settings: UnfoldingRunSettings):
    return PADUController(
        number_of_bs_antennas=system.array.bs_antennas,
        number_of_surface_elements=system.array.star_elements,
        number_of_users=system.number_of_users,
        history_length=system.learning.history_length,
        gru_hidden_size=run.architecture.gru_hidden_size,
        gru_number_of_layers=run.architecture.gru_number_of_layers,
        initializer_hidden_widths=settings.architecture.initializer_hidden_widths,
        unfolding_layers=settings.architecture.unfolding_layers,
        initial_primal_step_size=settings.architecture.initial_primal_step_size,
        initial_dual_step_size=settings.architecture.initial_dual_step_size,
        predictor_input_mode=settings.architecture.predictor_input_mode,
        design_payload_mode=settings.architecture.design_payload_mode,
        optimization_update_mode=(
            settings.architecture.optimization_update_mode
        ),
        csi_representation_mode=(
            settings.architecture.csi_representation_mode
        ),
        statistical_scenario_count=(
            settings.architecture.statistical_scenario_count
        ),
        statistical_scenario_seed=(
            settings.architecture.statistical_scenario_seed
        ),
        probabilistic_uncertainty_conditioning=(
            settings.architecture.probabilistic_uncertainty_conditioning
        ),
        controller_refinement_mode=(
            settings.architecture.controller_refinement_mode
        ),
        control_objective_mode=(
            settings.architecture.control_objective_mode
        ),
    )


def _build_context(
    system,
    audit: ExperimentAudit,
    epsilon: float,
    deployment_channel,
    device,
    settings: UnfoldingRunSettings | None = None,
):
    design_mode = (
        "business_minimum"
        if settings is None
        else settings.architecture.design_payload_mode
    )
    maximum_payload = (
        None
        if settings is None
        else settings.architecture.maximum_design_payload_bits
    )
    fixed_payload = (
        None
        if settings is None
        else settings.architecture.fixed_design_payload_bits
    )
    minimum_payloads = np.asarray(
        system.finite_blocklength.minimum_packet_payload_bits_per_user,
        dtype=np.float64,
    )
    if design_mode == "business_minimum":
        maximum_payloads = minimum_payloads.copy()
    elif design_mode == "fixed_common":
        if fixed_payload is None:
            raise ValueError("fixed design payload bits must be set")
        if any(fixed_payload < value for value in minimum_payloads):
            raise ValueError(
                "fixed design payload must not be below a business minimum"
            )
        maximum_payloads = np.full(
            system.number_of_users,
            fixed_payload,
            dtype=np.float64,
        )
    else:
        if maximum_payload is None:
            raise ValueError("maximum design payload bits must be set")
        if any(maximum_payload < value for value in minimum_payloads):
            raise ValueError(
                "maximum design payload must not be below a business minimum"
            )
        maximum_payloads = np.full(
            system.number_of_users,
            maximum_payload,
            dtype=np.float64,
        )
    payload_grid, sinr_grid = _design_sinr_lookup(
        system,
        minimum_payloads,
        maximum_payloads,
    )
    return PADUContext(
        bs_to_surface_channel=torch.from_numpy(
            np.asarray(deployment_channel, dtype=np.complex64)
        ).to(device),
        reflection_user_mask=torch.tensor(
            [side == "R" for side in system.mobility.user_sides],
            dtype=torch.bool,
            device=device,
        ),
        receiver_noise_power_watt=torch.full(
            (system.number_of_users,),
            audit.receiver_noise_power_watt,
            dtype=torch.float32,
            device=device,
        ),
        minimum_fbl_sinr=torch.tensor(
            audit.minimum_fbl_sinr_threshold_linear_per_user,
            dtype=torch.float32,
            device=device,
        ),
        minimum_design_payload_bits=torch.tensor(
            minimum_payloads,
            dtype=torch.float32,
            device=device,
        ),
        maximum_design_payload_bits=torch.tensor(
            maximum_payloads,
            dtype=torch.float32,
            device=device,
        ),
        design_payload_grid_bits=torch.tensor(
            payload_grid,
            dtype=torch.float32,
            device=device,
        ),
        design_sinr_grid=torch.tensor(
            sinr_grid,
            dtype=torch.float32,
            device=device,
        ),
        surface_noise_power_watt=audit.surface_noise_power_watt,
        bs_max_output_watt=audit.bs_power_budget_watt,
        star_total_max_output_watt=audit.surface_total_power_budget_watt,
        star_per_element_max_output_watt=audit.surface_per_element_power_budget_watt,
        star_max_power_gain=db_to_linear(system.power.star_max_power_gain_db),
        blocklength=system.finite_blocklength.blocklength,
        decoding_error_probability=system.finite_blocklength.decoding_error_probability,
        feasible_mapping_epsilon=epsilon,
        bs_per_antenna_max_output_watt=(
            None
            if audit.bs_per_antenna_power_budget_watt is None
            else torch.tensor(
                audit.bs_per_antenna_power_budget_watt,
                dtype=torch.float32,
                device=device,
            )
        ),
    )


def _design_sinr_lookup(
    system,
    minimum_payloads: np.ndarray,
    maximum_payloads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = float(np.min(minimum_payloads))
    upper = float(np.max(maximum_payloads))
    if upper == lower:
        upper = lower + 1.0
    payload_grid = np.linspace(lower, upper, 257, dtype=np.float64)
    threshold_grid = np.asarray(
        [
            finite_blocklength_rate_sinr_threshold(
                system.finite_blocklength.blocklength,
                system.finite_blocklength.decoding_error_probability,
                float(payload) / system.finite_blocklength.blocklength,
            )
            for payload in payload_grid
        ],
        dtype=np.float64,
    )
    return payload_grid, np.broadcast_to(
        threshold_grid,
        (system.number_of_users, threshold_grid.size),
    ).copy()


def _maximum_hardware_violation(controls, context):
    bs_power = torch.sum(torch.abs(controls.beamforming) ** 2, dim=(-2, -1))
    violations = [torch.relu(bs_power - context.bs_max_output_watt)]
    if context.bs_per_antenna_max_output_watt is not None:
        antenna_power = torch.sum(torch.abs(controls.beamforming) ** 2, dim=-1)
        violations.append(
            torch.amax(
                torch.relu(antenna_power - context.bs_per_antenna_max_output_watt),
                dim=-1,
            )
        )
    surface = surface_output_powers(
        context.bs_to_surface_channel,
        controls,
        context.surface_noise_power_watt,
    )
    violations.extend(
        (
            torch.relu(torch.sum(surface, dim=-1) - context.star_total_max_output_watt),
            torch.amax(
                torch.relu(surface - context.star_per_element_max_output_watt),
                dim=-1,
            ),
            torch.amax(torch.relu(1.0 - controls.power_gain), dim=-1),
            torch.amax(
                torch.relu(controls.power_gain - context.star_max_power_gain),
                dim=-1,
            ),
        )
    )
    return torch.amax(torch.stack(violations, dim=-1), dim=-1)


def _summarize_evaluation(rows, output: Path):
    domains = sorted({row["domain"] for row in rows})
    by_domain = {}
    for domain in domains:
        selected = [row for row in rows if row["domain"] == domain]
        by_domain[domain] = {
            "completed_slots": len(selected),
            "joint_qos_outage_count": int(sum(row["joint_qos_outage"] for row in selected)),
            "joint_qos_outage_rate": float(np.mean([row["joint_qos_outage"] for row in selected])),
            "mean_actual_weighted_throughput_bps": float(np.mean([row["actual_weighted_throughput_bps"] for row in selected])),
            "mean_prediction_nmse": float(np.mean([row["prediction_nmse"] for row in selected])),
            "mean_control_time_s": float(np.mean([row["control_total_s"] for row in selected])),
            "maximum_constraint_violation": float(max(row["maximum_constraint_violation"] for row in selected)),
            "mean_design_payload_bits": float(np.mean([row["mean_design_payload_bits"] for row in selected])),
            "minimum_design_payload_bits": float(min(row["minimum_design_payload_bits"] for row in selected)),
            "maximum_design_payload_bits": float(max(row["maximum_design_payload_bits"] for row in selected)),
        }
    return {"domains": by_domain, "output_directory": str(output)}


def _model_dimensions(system, run, settings):
    dimensions = {
        "number_of_bs_antennas": system.array.bs_antennas,
        "number_of_surface_elements": system.array.star_elements,
        "number_of_users": system.number_of_users,
        "history_length": system.learning.history_length,
        "gru_hidden_size": run.architecture.gru_hidden_size,
        "gru_number_of_layers": run.architecture.gru_number_of_layers,
        "unfolding_layers": settings.architecture.unfolding_layers,
        "initializer_hidden_widths": list(settings.architecture.initializer_hidden_widths),
    }
    if settings.architecture.predictor_input_mode != "channel_only":
        dimensions["predictor_input_mode"] = (
            settings.architecture.predictor_input_mode
        )
    if settings.architecture.design_payload_mode != "business_minimum":
        dimensions["design_payload_mode"] = (
            settings.architecture.design_payload_mode
        )
        if settings.architecture.design_payload_mode in {
            "learned_userwise",
            "learned_userwise_causal_variation",
        }:
            dimensions["maximum_design_payload_bits"] = (
                settings.architecture.maximum_design_payload_bits
            )
        else:
            dimensions["fixed_design_payload_bits"] = (
                settings.architecture.fixed_design_payload_bits
            )
    if settings.architecture.optimization_update_mode != "standard":
        dimensions["optimization_update_mode"] = (
            settings.architecture.optimization_update_mode
        )
    if settings.architecture.csi_representation_mode != "gru_point":
        dimensions["csi_representation_mode"] = (
            settings.architecture.csi_representation_mode
        )
        dimensions["statistical_scenario_count"] = (
            settings.architecture.statistical_scenario_count
        )
        dimensions["statistical_scenario_seed"] = (
            settings.architecture.statistical_scenario_seed
        )
        if (
            settings.architecture.csi_representation_mode == "gru_probabilistic"
            and not settings.architecture.probabilistic_uncertainty_conditioning
        ):
            dimensions["probabilistic_uncertainty_conditioning"] = False
    if settings.architecture.control_objective_mode != "sum_rate":
        dimensions["control_objective_mode"] = (
            settings.architecture.control_objective_mode
        )
    if (
        settings.architecture.controller_refinement_mode
        != "primal_dual_unfolding"
    ):
        dimensions["controller_refinement_mode"] = (
            settings.architecture.controller_refinement_mode
        )
    return dimensions


def _design_payload_row(design_payload_bits: Tensor) -> dict[str, float]:
    values = design_payload_bits.detach().cpu().tolist()
    return {
        "minimum_design_payload_bits": float(min(values)),
        "maximum_design_payload_bits": float(max(values)),
        "mean_design_payload_bits": float(np.mean(values)),
        **{
            f"design_payload_bits_user_{index + 1}": float(value)
            for index, value in enumerate(values)
        },
    }


def _causal_variation_row(causal_variation: Tensor) -> dict[str, float]:
    values = causal_variation.detach().cpu().tolist()
    return {
        "mean_causal_channel_variation": float(np.mean(values)),
        **{
            f"causal_channel_variation_user_{index + 1}": float(value)
            for index, value in enumerate(values)
        },
    }


def _innovation_scale_row(
    values: Tensor | None,
    batch_index: int,
) -> dict[str, float]:
    if values is None:
        return {}
    selected = values[batch_index].detach().cpu().tolist()
    return {
        "mean_conditional_innovation_standard_deviation": float(
            np.mean(selected)
        ),
        **{
            "conditional_innovation_standard_deviation_user_"
            f"{index + 1}": float(value)
            for index, value in enumerate(selected)
        },
    }


def _task_channels(tasks):
    return [task.trajectory.surface_to_user_channels for task in tasks]


def _generate_selected_task_families(
    *,
    system,
    run,
    deployment,
    streams: dict[str, int],
    family_names: Sequence[str],
):
    specifications = {
        "gru_training": (
            "training",
            run.dataset.gru_training_tasks,
            "gru-train",
            "gru_training_tasks",
        ),
        "gru_validation": (
            "validation",
            run.dataset.gru_validation_tasks,
            "gru-validation",
            "gru_validation_tasks",
        ),
        "meta_training": (
            "training",
            run.dataset.meta_training_tasks,
            "meta-train",
            "meta_training_tasks",
        ),
        "meta_validation": (
            "validation",
            run.dataset.meta_validation_tasks,
            "meta-validation",
            "meta_validation_tasks",
        ),
        "in_domain_test": (
            "in_domain_test",
            run.dataset.in_domain_test_tasks,
            "in-domain-test",
            "in_domain_test_tasks",
        ),
        "out_of_domain_test": (
            "out_of_domain_test",
            run.dataset.out_of_domain_test_tasks,
            "out-of-domain-test",
            "out_of_domain_test_tasks",
        ),
    }
    requested = tuple(family_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("family_names must be non-empty and unique")
    unknown = sorted(set(requested) - set(specifications))
    if unknown:
        raise ValueError(f"family_names contains unknown values: {unknown}")
    output = {}
    for family_name in requested:
        distribution_name, count, prefix, stream_name = specifications[family_name]
        output[family_name] = generate_task_family(
            config=system,
            deployment=deployment,
            distribution=run.task_distributions[distribution_name],
            number_of_tasks=count,
            number_of_slots=run.dataset.trajectory_slots,
            task_id_prefix=prefix,
            rng=np.random.default_rng(streams[stream_name]),
        )
    return output


def _data_loader(dataset, batch_size, shuffle, workers, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def _preloaded_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
):
    """Build a zero-copy training loader from tensors already on the device."""
    if device.type != "cuda":
        raise ValueError("preloaded training tensors require a CUDA device")
    if len(dataset) == 0:
        raise ValueError("dataset must be non-empty")
    columns = tuple(
        torch.stack(
            [dataset[index][column] for index in range(len(dataset))]
        ).to(device=device, non_blocking=True)
        for column in range(len(dataset[0]))
    )
    preloaded = TensorDataset(*columns)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        preloaded,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _object(value: Any, location: str):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], location: str):
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{location} keys are invalid; missing={missing}, unknown={unknown}"
        )


def _positive_int(value: Any, location: str):
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return int(value)


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be boolean")
    return value


def _nonnegative_int(value: Any, location: str):
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return int(value)


def _positive_float(value: Any, location: str):
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{location} must be finite and positive")
    return number


def _nonnegative_float(value: Any, location: str):
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{location} must be finite and non-negative")
    return number


def _optional_positive_float(value: Any, location: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, location)


def _predictor_input_mode(value: Any) -> str:
    allowed = {
        "channel_only",
        "channel_and_first_difference",
    }
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "architecture.predictor_input_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _design_payload_mode(value: Any) -> str:
    allowed = {
        "business_minimum",
        "fixed_common",
        "learned_userwise",
        "learned_userwise_causal_variation",
    }
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "architecture.design_payload_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _optimization_update_mode(value: Any) -> str:
    allowed = {"standard", "user_conditioned_dual_momentum"}
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "architecture.optimization_update_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _control_objective_mode(value: Any) -> str:
    allowed = {"sum_rate", "max_min_normalized_payload"}
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "architecture.control_objective_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _csi_representation_mode(value: Any) -> str:
    allowed = {
        "gru_point",
        "gru_probabilistic",
        "latest_observed_csi",
    }
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "architecture.csi_representation_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _controller_refinement_mode(value: Any) -> str:
    allowed = {"primal_dual_unfolding", "initializer_only"}
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "architecture.controller_refinement_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _joint_qos_loss_mode(value: Any) -> str:
    allowed = {
        "smooth_payload_boundary",
        "squared_max_sinr_shortfall",
    }
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "training.joint_qos_loss_mode must equal one of "
            f"{sorted(allowed)}"
        )
    return value


def _training_strategy(value: Any) -> str:
    allowed = {
        "joint",
        "predictor_then_controller",
        "controller_only",
    }
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            "training.training_strategy must equal one of "
            f"{sorted(allowed)}"
        )
    return value
