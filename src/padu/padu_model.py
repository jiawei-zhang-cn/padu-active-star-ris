"""PADU service-shortfall-guided unfolding model for active STAR-RIS control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import IncrementalSharedGRUPredictor
from .torch_physics import (
    TorchControls,
    decode_feasible_controls,
    torch_fbl_spectral_efficiency,
    torch_user_sinr,
)


@dataclass(frozen=True)
class PADUContext:
    bs_to_surface_channel: Tensor
    reflection_user_mask: Tensor
    receiver_noise_power_watt: Tensor
    minimum_fbl_sinr: Tensor
    minimum_design_payload_bits: Tensor
    maximum_design_payload_bits: Tensor
    design_payload_grid_bits: Tensor
    design_sinr_grid: Tensor
    surface_noise_power_watt: float
    bs_max_output_watt: float
    star_total_max_output_watt: float
    star_per_element_max_output_watt: float
    star_max_power_gain: float
    blocklength: int
    decoding_error_probability: float
    feasible_mapping_epsilon: float
    bs_per_antenna_max_output_watt: Tensor | None = None


@dataclass(frozen=True)
class PADUOutput:
    predicted_channel: Tensor
    controls: TorchControls
    layer_controls: tuple[TorchControls, ...]
    layer_sinr: tuple[Tensor, ...]
    layer_scenario_sinr: tuple[Tensor, ...]
    dual_variables: Tensor
    design_payload_bits: Tensor
    design_sinr_thresholds: Tensor
    causal_channel_variation: Tensor
    conditional_innovation_standard_deviation: Tensor | None = None


class PADUController(nn.Module):
    """Incremental channel prediction followed by fixed-depth control updates."""

    def __init__(
        self,
        *,
        number_of_bs_antennas: int,
        number_of_surface_elements: int,
        number_of_users: int,
        history_length: int,
        gru_hidden_size: int,
        gru_number_of_layers: int,
        initializer_hidden_widths: Sequence[int],
        unfolding_layers: int,
        initial_primal_step_size: float,
        initial_dual_step_size: float,
        predictor_input_mode: str = "channel_only",
        csi_representation_mode: str = "gru_point",
        statistical_scenario_count: int = 1,
        statistical_scenario_seed: int | None = None,
        probabilistic_uncertainty_conditioning: bool = False,
        controller_refinement_mode: str = "primal_dual_unfolding",
        design_payload_mode: str = "business_minimum",
        optimization_update_mode: str = "standard",
        control_objective_mode: str = "sum_rate",
    ) -> None:
        super().__init__()
        dimensions = (
            number_of_bs_antennas,
            number_of_surface_elements,
            number_of_users,
            history_length,
            unfolding_layers,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all system and unfolding dimensions must be positive")
        if not initializer_hidden_widths or any(
            width <= 0 for width in initializer_hidden_widths
        ):
            raise ValueError("initializer_hidden_widths must be positive")
        if initial_primal_step_size <= 0.0 or initial_dual_step_size <= 0.0:
            raise ValueError("initial step sizes must be positive")

        self.number_of_bs_antennas = number_of_bs_antennas
        self.number_of_surface_elements = number_of_surface_elements
        self.number_of_users = number_of_users
        self.history_length = history_length
        self.unfolding_layers = unfolding_layers
        self.predictor_input_mode = predictor_input_mode
        if csi_representation_mode not in {
            "gru_point",
            "gru_probabilistic",
            "latest_observed_csi",
        }:
            raise ValueError("csi_representation_mode is unsupported")
        if statistical_scenario_count != 1:
            raise ValueError("statistical_scenario_count must equal 1")
        if (
            csi_representation_mode == "gru_point"
            and statistical_scenario_count != 1
        ):
            raise ValueError("GRU point CSI requires one scenario")
        self.csi_representation_mode = csi_representation_mode
        if controller_refinement_mode not in {
            "primal_dual_unfolding",
            "initializer_only",
        }:
            raise ValueError("controller_refinement_mode is unsupported")
        self.controller_refinement_mode = controller_refinement_mode
        self.statistical_scenario_count = statistical_scenario_count
        if not isinstance(probabilistic_uncertainty_conditioning, bool):
            raise ValueError(
                "probabilistic_uncertainty_conditioning must be boolean"
            )
        self.probabilistic_uncertainty_conditioning = (
            probabilistic_uncertainty_conditioning
        )
        if statistical_scenario_seed is not None:
            raise ValueError("statistical_scenario_seed must be omitted")
        directions = torch.empty(0, dtype=torch.complex64)
        self.register_buffer(
            "statistical_scenario_directions",
            directions,
            persistent=False,
        )
        if design_payload_mode not in {
            "business_minimum",
            "fixed_common",
            "learned_userwise",
            "learned_userwise_causal_variation",
        }:
            raise ValueError("design_payload_mode is unsupported")
        self.design_payload_mode = design_payload_mode
        if optimization_update_mode not in {
            "standard",
            "user_conditioned_dual_momentum",
        }:
            raise ValueError("optimization_update_mode is unsupported")
        self.optimization_update_mode = optimization_update_mode
        if control_objective_mode not in {
            "sum_rate",
            "max_min_normalized_payload",
        }:
            raise ValueError("control_objective_mode is unsupported")
        self.control_objective_mode = control_objective_mode
        self.raw_control_size = (
            2 * number_of_bs_antennas * number_of_users
            + 4 * number_of_surface_elements
        )
        self.raw_beam_size = 2 * number_of_bs_antennas * number_of_users
        self.predictor = (
            None
            if csi_representation_mode == "latest_observed_csi"
            else IncrementalSharedGRUPredictor(
                number_of_surface_elements=number_of_surface_elements,
                hidden_size=gru_hidden_size,
                number_of_layers=gru_number_of_layers,
                input_mode=predictor_input_mode,
                predict_distribution_scale=(
                    csi_representation_mode == "gru_probabilistic"
                ),
            )
        )
        self.probabilistic_uncertainty_encoder = (
            nn.Linear(1, gru_hidden_size)
            if (
                csi_representation_mode == "gru_probabilistic"
                and probabilistic_uncertainty_conditioning
            )
            else None
        )
        self.design_margin_head = (
            None
            if design_payload_mode == "business_minimum"
            else nn.Linear(gru_hidden_size, 1)
        )
        if self.design_margin_head is not None:
            nn.init.zeros_(self.design_margin_head.weight)
            nn.init.zeros_(self.design_margin_head.bias)
            if design_payload_mode == "fixed_common":
                self.design_margin_head.requires_grad_(False)
        self.raw_causal_variation_scale = (
            nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
            if design_payload_mode
            == "learned_userwise_causal_variation"
            else None
        )

        initializer_input = (
            2 * number_of_surface_elements * number_of_users
            + gru_hidden_size
        )
        layers: list[nn.Module] = []
        previous = initializer_input
        for width in initializer_hidden_widths:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, self.raw_control_size))
        self.control_initializer = nn.Sequential(*layers)

        refinement_layers = (
            unfolding_layers
            if controller_refinement_mode == "primal_dual_unfolding"
            else 0
        )
        self.step_conditioners = nn.ModuleList(
            nn.Linear(gru_hidden_size, 5) for _ in range(refinement_layers)
        )
        primal_raw = _inverse_softplus(initial_primal_step_size)
        dual_raw = _inverse_softplus(initial_dual_step_size)
        self.raw_beam_step_sizes = nn.Parameter(
            torch.full((refinement_layers,), primal_raw)
        )
        self.raw_shared_surface_step_sizes = nn.Parameter(
            torch.full((refinement_layers,), primal_raw)
        )
        self.raw_reflection_phase_step_sizes = nn.Parameter(
            torch.full((refinement_layers,), primal_raw)
        )
        self.raw_transmission_phase_step_sizes = nn.Parameter(
            torch.full((refinement_layers,), primal_raw)
        )
        self.raw_dual_step_sizes = nn.Parameter(
            torch.full((refinement_layers,), dual_raw)
        )
        if optimization_update_mode == "user_conditioned_dual_momentum":
            self.dual_initializer = nn.Linear(gru_hidden_size, 1)
            self.dual_step_conditioners = nn.ModuleList(
                nn.Linear(gru_hidden_size, 1)
                for _ in range(refinement_layers)
            )
            nn.init.zeros_(self.dual_initializer.weight)
            nn.init.zeros_(self.dual_initializer.bias)
            for conditioner in self.dual_step_conditioners:
                nn.init.zeros_(conditioner.weight)
                nn.init.zeros_(conditioner.bias)
            self.raw_beam_momentum = nn.Parameter(
                torch.zeros(refinement_layers)
            )
            self.raw_surface_momentum = nn.Parameter(
                torch.zeros(refinement_layers)
            )
        else:
            self.dual_initializer = None
            self.dual_step_conditioners = None
            self.register_parameter("raw_beam_momentum", None)
            self.register_parameter("raw_surface_momentum", None)

    def forward(
        self,
        normalized_history: Tensor,
        normalizer_mean: Tensor,
        normalizer_standard_deviation: Tensor,
        context: PADUContext,
        perfect_next_slot_channel: Tensor | None = None,
        conditional_mean_channel: Tensor | None = None,
        conditional_innovation_standard_deviation: Tensor | None = None,
        phase_boundary: Callable[[str], None] | None = None,
    ) -> PADUOutput:
        if phase_boundary is not None:
            phase_boundary("prediction_start")
        expected = (
            self.number_of_users,
            self.history_length,
            2 * self.number_of_surface_elements,
        )
        if normalized_history.ndim != 4 or normalized_history.shape[1:] != expected:
            raise ValueError(
                "normalized_history must have shape (batch, K, history, 2N)"
            )
        feature_size = 2 * self.number_of_surface_elements
        if normalizer_mean.shape != (feature_size,):
            raise ValueError("normalizer_mean must have shape (2N,)")
        if normalizer_standard_deviation.shape != (feature_size,):
            raise ValueError(
                "normalizer_standard_deviation must have shape (2N,)"
            )

        batch_size = normalized_history.shape[0]
        if self.csi_representation_mode in {
            "gru_point",
            "gru_probabilistic",
        }:
            if self.predictor is None:
                raise RuntimeError("GRU predictor is missing")
            if self.csi_representation_mode == "gru_probabilistic":
                (
                    predicted_channel,
                    normalized_prediction,
                    user_hidden,
                    normalized_innovation_standard_deviation,
                ) = self.predict_channel_distribution(
                    normalized_history,
                    normalizer_mean,
                    normalizer_standard_deviation,
                )
                if self.probabilistic_uncertainty_conditioning:
                    if self.probabilistic_uncertainty_encoder is None:
                        raise RuntimeError(
                            "probabilistic uncertainty encoder is missing"
                        )
                    user_hidden = user_hidden + (
                        self.probabilistic_uncertainty_encoder(
                            torch.log(
                                normalized_innovation_standard_deviation
                            ).unsqueeze(-1)
                        )
                    )
                if self.statistical_scenario_count == 1:
                    scenario_channels = predicted_channel.unsqueeze(1)
                conditional_innovation_standard_deviation = (
                    _physical_equivalent_innovation_standard_deviation(
                        normalized_innovation_standard_deviation,
                        normalizer_standard_deviation,
                    )
                )
            else:
                predicted_channel, normalized_prediction, user_hidden = (
                    self.predict_channel(
                        normalized_history,
                        normalizer_mean,
                        normalizer_standard_deviation,
                    )
                )
                scenario_channels = predicted_channel.unsqueeze(1)
        else:
            normalized_prediction = normalized_history[:, :, -1, :]
            predicted_channel = _normalized_features_to_channel(
                normalized_prediction,
                normalizer_mean,
                normalizer_standard_deviation,
            )
            user_hidden = torch.zeros(
                batch_size,
                self.number_of_users,
                self.control_initializer[0].in_features
                - 2 * self.number_of_surface_elements * self.number_of_users,
                dtype=normalized_history.dtype,
                device=normalized_history.device,
            )
            scenario_channels = predicted_channel.unsqueeze(1)
        if perfect_next_slot_channel is not None:
            expected_channel_shape = (
                batch_size,
                self.number_of_surface_elements,
                self.number_of_users,
            )
            if perfect_next_slot_channel.shape != expected_channel_shape:
                raise ValueError(
                    "perfect_next_slot_channel must have shape (batch, N, K)"
                )
            if not torch.is_complex(perfect_next_slot_channel):
                raise ValueError("perfect_next_slot_channel must be complex")
            predicted_channel = perfect_next_slot_channel
            normalized_prediction = _channel_to_normalized_features(
                predicted_channel,
                normalizer_mean,
                normalizer_standard_deviation,
            )
            scenario_channels = predicted_channel.unsqueeze(1)
        if phase_boundary is not None:
            phase_boundary("prediction_end")
        causal_variation = _causal_channel_variation_from_normalized_history(
            normalized_history,
            normalizer_mean,
            normalizer_standard_deviation,
        )
        controller_condition = torch.mean(user_hidden, dim=1)
        initializer_input = torch.cat(
            (
                normalized_prediction.reshape(batch_size, -1),
                controller_condition,
            ),
            dim=-1,
        )
        if phase_boundary is not None:
            phase_boundary("initializer_start")
        raw_controls = self.control_initializer(initializer_input)
        if phase_boundary is not None:
            phase_boundary("initializer_end")
            phase_boundary("refinement_start")
        design_payload_bits, design_sinr_thresholds = self._design_targets(
            user_hidden,
            causal_variation,
            context,
        )
        if self.dual_initializer is None:
            dual = torch.zeros(
                batch_size,
                self.number_of_users,
                dtype=normalized_history.dtype,
                device=normalized_history.device,
            )
        else:
            dual = F.softplus(
                self.dual_initializer(user_hidden).squeeze(-1)
            )
        beam_momentum = torch.zeros_like(raw_controls)
        surface_momentum = torch.zeros_like(raw_controls)

        layer_controls: list[TorchControls] = []
        layer_sinr: list[Tensor] = []
        layer_scenario_sinr: list[Tensor] = []
        if self.controller_refinement_mode == "initializer_only":
            controls = self._decode(raw_controls, context)
            scenario_sinr = _scenario_user_sinr(
                bs_to_surface_channel=context.bs_to_surface_channel,
                scenario_channels=scenario_channels,
                controls=controls,
                reflection_user_mask=context.reflection_user_mask,
                surface_noise_power_watt=context.surface_noise_power_watt,
                receiver_noise_power_watt=context.receiver_noise_power_watt,
            )
            layer_controls.append(controls)
            layer_sinr.append(torch.mean(scenario_sinr, dim=1))
            layer_scenario_sinr.append(scenario_sinr)
        for layer_index in range(self.unfolding_layers):
            if self.controller_refinement_mode == "initializer_only":
                break
            current_controls = self._decode(raw_controls, context)
            current_sinr_scenarios = _scenario_user_sinr(
                bs_to_surface_channel=context.bs_to_surface_channel,
                scenario_channels=scenario_channels,
                controls=current_controls,
                reflection_user_mask=context.reflection_user_mask,
                surface_noise_power_watt=context.surface_noise_power_watt,
                receiver_noise_power_watt=context.receiver_noise_power_watt,
            )
            current_sinr = torch.mean(current_sinr_scenarios, dim=1)
            current_deficit = _normalized_sinr_shortfall(
                torch.amin(current_sinr_scenarios, dim=1),
                design_sinr_thresholds,
            )
            gates = 0.5 + torch.sigmoid(
                self.step_conditioners[layer_index](controller_condition)
            )
            beam_step = (
                F.softplus(self.raw_beam_step_sizes[layer_index]) * gates[:, 0]
            )
            shared_surface_step = (
                F.softplus(self.raw_shared_surface_step_sizes[layer_index])
                * gates[:, 1]
            )
            reflection_phase_step = (
                F.softplus(
                    self.raw_reflection_phase_step_sizes[layer_index]
                )
                * gates[:, 2]
            )
            transmission_phase_step = (
                F.softplus(
                    self.raw_transmission_phase_step_sizes[layer_index]
                )
                * gates[:, 3]
            )
            if self.dual_step_conditioners is None:
                dual_step = (
                    F.softplus(self.raw_dual_step_sizes[layer_index])
                    * gates[:, 4, None]
                )
            else:
                dual_step = F.softplus(
                    self.raw_dual_step_sizes[layer_index]
                ) * (
                    0.5
                    + torch.sigmoid(
                        self.dual_step_conditioners[layer_index](
                            user_hidden
                        ).squeeze(-1)
                    )
                )
            dual = torch.relu(dual + dual_step * current_deficit)

            raw_controls, beam_momentum = self._primal_update(
                raw_controls=raw_controls,
                scenario_channels=scenario_channels,
                dual=dual,
                design_sinr_thresholds=design_sinr_thresholds,
                context=context,
                update_slices=(slice(0, self.raw_beam_size),),
                step_sizes=(beam_step,),
                previous_momentum=beam_momentum,
                momentum_coefficient=(
                    None
                    if self.raw_beam_momentum is None
                    else torch.sigmoid(
                        self.raw_beam_momentum[layer_index]
                    )
                ),
            )
            shared_start = self.raw_beam_size
            reflection_phase_start = (
                shared_start + 2 * self.number_of_surface_elements
            )
            transmission_phase_start = (
                reflection_phase_start + self.number_of_surface_elements
            )
            raw_controls, surface_momentum = self._primal_update(
                raw_controls=raw_controls,
                scenario_channels=scenario_channels,
                dual=dual,
                design_sinr_thresholds=design_sinr_thresholds,
                context=context,
                update_slices=(
                    slice(shared_start, reflection_phase_start),
                    slice(reflection_phase_start, transmission_phase_start),
                    slice(transmission_phase_start, self.raw_control_size),
                ),
                step_sizes=(
                    shared_surface_step,
                    reflection_phase_step,
                    transmission_phase_step,
                ),
                previous_momentum=surface_momentum,
                momentum_coefficient=(
                    None
                    if self.raw_surface_momentum is None
                    else torch.sigmoid(
                        self.raw_surface_momentum[layer_index]
                    )
                ),
            )
            controls = self._decode(raw_controls, context)
            scenario_sinr = _scenario_user_sinr(
                bs_to_surface_channel=context.bs_to_surface_channel,
                scenario_channels=scenario_channels,
                controls=controls,
                reflection_user_mask=context.reflection_user_mask,
                surface_noise_power_watt=context.surface_noise_power_watt,
                receiver_noise_power_watt=context.receiver_noise_power_watt,
            )
            sinr = torch.mean(scenario_sinr, dim=1)
            layer_controls.append(controls)
            layer_sinr.append(sinr)
            layer_scenario_sinr.append(scenario_sinr)

        output = PADUOutput(
            predicted_channel=predicted_channel,
            controls=layer_controls[-1],
            layer_controls=tuple(layer_controls),
            layer_sinr=tuple(layer_sinr),
            layer_scenario_sinr=tuple(layer_scenario_sinr),
            dual_variables=dual,
            design_payload_bits=design_payload_bits,
            design_sinr_thresholds=design_sinr_thresholds,
            causal_channel_variation=causal_variation,
            conditional_innovation_standard_deviation=(
                conditional_innovation_standard_deviation
            ),
        )
        if phase_boundary is not None:
            phase_boundary("refinement_end")
        return output

    def predict_channel(
        self,
        normalized_history: Tensor,
        normalizer_mean: Tensor,
        normalizer_standard_deviation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        feature_size = 2 * self.number_of_surface_elements
        expected = (
            self.number_of_users,
            self.history_length,
            feature_size,
        )
        if normalized_history.ndim != 4 or normalized_history.shape[1:] != expected:
            raise ValueError(
                "normalized_history must have shape (batch, K, history, 2N)"
            )
        if normalizer_mean.shape != (feature_size,):
            raise ValueError("normalizer_mean must have shape (2N,)")
        if normalizer_standard_deviation.shape != (feature_size,):
            raise ValueError(
                "normalizer_standard_deviation must have shape (2N,)"
            )
        batch_size = normalized_history.shape[0]
        if self.predictor is None:
            raise RuntimeError("GRU predictor is unavailable")
        predictor_history = _predictor_history_features(
            normalized_history,
            self.predictor_input_mode,
        )
        flattened_history = predictor_history.reshape(
            batch_size * self.number_of_users,
            self.history_length,
            predictor_history.shape[-1],
        )
        normalized_prediction, user_hidden = self.predictor(flattened_history)
        normalized_prediction = normalized_prediction.reshape(
            batch_size, self.number_of_users, feature_size
        )
        user_hidden = user_hidden.reshape(
            batch_size, self.number_of_users, -1
        )
        predicted_channel = _normalized_features_to_channel(
            normalized_prediction,
            normalizer_mean,
            normalizer_standard_deviation,
        )
        return predicted_channel, normalized_prediction, user_hidden

    def predict_normalized_innovation_standard_deviation(
        self,
        normalized_history: Tensor,
    ) -> Tensor:
        if self.csi_representation_mode != "gru_probabilistic":
            raise RuntimeError(
                "normalized innovation scale requires probabilistic GRU CSI"
            )
        if self.predictor is None:
            raise RuntimeError("GRU predictor is unavailable")
        batch_size = normalized_history.shape[0]
        predictor_history = _predictor_history_features(
            normalized_history,
            self.predictor_input_mode,
        )
        flattened_history = predictor_history.reshape(
            batch_size * self.number_of_users,
            self.history_length,
            predictor_history.shape[-1],
        )
        _, _, normalized_standard_deviation = self.predictor.distribution(
            flattened_history
        )
        return normalized_standard_deviation.reshape(
            batch_size, self.number_of_users
        )

    def predict_channel_distribution(
        self,
        normalized_history: Tensor,
        normalizer_mean: Tensor,
        normalizer_standard_deviation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict the next normalized mean and one scale per user."""
        if self.csi_representation_mode != "gru_probabilistic":
            raise RuntimeError(
                "channel distribution requires probabilistic GRU CSI"
            )
        if self.predictor is None:
            raise RuntimeError("GRU predictor is unavailable")
        feature_size = 2 * self.number_of_surface_elements
        expected = (
            self.number_of_users,
            self.history_length,
            feature_size,
        )
        if normalized_history.ndim != 4 or normalized_history.shape[1:] != expected:
            raise ValueError(
                "normalized_history must have shape (batch, K, history, 2N)"
            )
        if normalizer_mean.shape != (feature_size,):
            raise ValueError("normalizer_mean must have shape (2N,)")
        if normalizer_standard_deviation.shape != (feature_size,):
            raise ValueError(
                "normalizer_standard_deviation must have shape (2N,)"
            )
        batch_size = normalized_history.shape[0]
        predictor_history = _predictor_history_features(
            normalized_history,
            self.predictor_input_mode,
        )
        flattened_history = predictor_history.reshape(
            batch_size * self.number_of_users,
            self.history_length,
            predictor_history.shape[-1],
        )
        (
            normalized_prediction,
            user_hidden,
            normalized_standard_deviation,
        ) = self.predictor.distribution(flattened_history)
        normalized_prediction = normalized_prediction.reshape(
            batch_size, self.number_of_users, feature_size
        )
        user_hidden = user_hidden.reshape(
            batch_size, self.number_of_users, -1
        )
        normalized_standard_deviation = (
            normalized_standard_deviation.reshape(
                batch_size, self.number_of_users
            )
        )
        predicted_channel = _normalized_features_to_channel(
            normalized_prediction,
            normalizer_mean,
            normalizer_standard_deviation,
        )
        return (
            predicted_channel,
            normalized_prediction,
            user_hidden,
            normalized_standard_deviation,
        )

    def _primal_update(
        self,
        *,
        raw_controls: Tensor,
        scenario_channels: Tensor,
        dual: Tensor,
        design_sinr_thresholds: Tensor,
        context: PADUContext,
        update_slices: tuple[slice, ...],
        step_sizes: tuple[Tensor, ...],
        previous_momentum: Tensor,
        momentum_coefficient: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if len(update_slices) != len(step_sizes) or not update_slices:
            raise ValueError("update slices and step sizes must align")
        with torch.enable_grad():
            if not raw_controls.requires_grad:
                raw_controls = raw_controls.requires_grad_(True)
            controls = self._decode(raw_controls, context)
            sinr_scenarios = _scenario_user_sinr(
                bs_to_surface_channel=context.bs_to_surface_channel,
                scenario_channels=scenario_channels,
                controls=controls,
                reflection_user_mask=context.reflection_user_mask,
                surface_noise_power_watt=context.surface_noise_power_watt,
                receiver_noise_power_watt=context.receiver_noise_power_watt,
            )
            scenario_rate = torch_fbl_spectral_efficiency(
                sinr_scenarios,
                blocklength=context.blocklength,
                decoding_error_probability=(
                    context.decoding_error_probability
                ),
            )
            worst_sinr = torch.amin(sinr_scenarios, dim=1)
            deficit = _normalized_sinr_shortfall(
                worst_sinr, design_sinr_thresholds
            )
            if self.control_objective_mode == "sum_rate":
                utility = torch.sum(
                    torch.mean(scenario_rate, dim=1), dim=-1
                )
            else:
                minimum_payload = (
                    context.minimum_design_payload_bits.to(
                        dtype=scenario_rate.dtype,
                        device=scenario_rate.device,
                    )
                )
                if torch.any(minimum_payload <= 0.0):
                    raise ValueError(
                        "max-min normalized payload requires positive "
                        "business payloads"
                    )
                normalized_payload = (
                    context.blocklength
                    * scenario_rate
                    / minimum_payload[None, None, :]
                )
                utility = torch.amin(
                    normalized_payload, dim=(1, 2)
                )
            lagrangian = torch.sum(
                utility - torch.sum(dual * deficit, dim=-1)
            )
            gradient = torch.autograd.grad(
                lagrangian,
                raw_controls,
                create_graph=self.training,
            )[0]
        update = torch.zeros_like(raw_controls)
        next_momentum = previous_momentum.clone()
        for update_slice, step_size in zip(
            update_slices, step_sizes, strict=True
        ):
            selected = gradient[:, update_slice]
            normalized = selected / torch.sqrt(
                torch.mean(selected.square(), dim=-1, keepdim=True)
                + 1.0e-12
            )
            if momentum_coefficient is None:
                selected_momentum = normalized
            else:
                selected_momentum = (
                    momentum_coefficient
                    * previous_momentum[:, update_slice]
                    + (1.0 - momentum_coefficient) * normalized
                )
            next_momentum[:, update_slice] = selected_momentum
            update[:, update_slice] = (
                step_size[:, None] * selected_momentum
            )
        return raw_controls + update, next_momentum

    def _design_targets(
        self,
        user_hidden: Tensor,
        causal_channel_variation: Tensor,
        context: PADUContext,
    ) -> tuple[Tensor, Tensor]:
        minimum_payload = context.minimum_design_payload_bits.to(
            dtype=user_hidden.dtype,
            device=user_hidden.device,
        )
        maximum_payload = context.maximum_design_payload_bits.to(
            dtype=user_hidden.dtype,
            device=user_hidden.device,
        )
        if minimum_payload.shape != (self.number_of_users,):
            raise ValueError("minimum design payload must have shape (K,)")
        if maximum_payload.shape != (self.number_of_users,):
            raise ValueError("maximum design payload must have shape (K,)")
        if torch.any(maximum_payload < minimum_payload):
            raise ValueError(
                "maximum design payload must not be below the business minimum"
            )
        if self.design_payload_mode == "business_minimum":
            payload = minimum_payload.unsqueeze(0).expand(
                user_hidden.shape[0], -1
            )
            sinr = context.minimum_fbl_sinr.to(
                dtype=user_hidden.dtype,
                device=user_hidden.device,
            ).unsqueeze(0).expand(user_hidden.shape[0], -1)
            return payload, sinr
        if self.design_payload_mode == "fixed_common":
            payload = maximum_payload.unsqueeze(0).expand(
                user_hidden.shape[0], -1
            )
            return payload, _interpolate_design_sinr(payload, context)
        if self.design_margin_head is None:
            raise RuntimeError("learned design payload head is missing")
        margin_logit = self.design_margin_head(user_hidden).squeeze(-1)
        if self.design_payload_mode == "learned_userwise_causal_variation":
            if self.raw_causal_variation_scale is None:
                raise RuntimeError("causal variation scale is missing")
            if causal_channel_variation.shape != margin_logit.shape:
                raise ValueError(
                    "causal channel variation must have shape (batch, K)"
                )
            margin_logit = margin_logit + F.softplus(
                self.raw_causal_variation_scale
            ) * torch.log1p(causal_channel_variation)
        payload = minimum_payload + torch.sigmoid(margin_logit) * (
            maximum_payload - minimum_payload
        )
        return payload, _interpolate_design_sinr(payload, context)

    def _decode(
        self,
        raw_controls: Tensor,
        context: PADUContext,
    ) -> TorchControls:
        return decode_feasible_controls(
            raw_output=raw_controls,
            bs_to_surface_channel=context.bs_to_surface_channel,
            number_of_bs_antennas=self.number_of_bs_antennas,
            number_of_surface_elements=self.number_of_surface_elements,
            number_of_users=self.number_of_users,
            bs_max_output_watt=context.bs_max_output_watt,
            star_total_max_output_watt=context.star_total_max_output_watt,
            star_per_element_max_output_watt=(
                context.star_per_element_max_output_watt
            ),
            star_max_power_gain=context.star_max_power_gain,
            surface_noise_power_watt=context.surface_noise_power_watt,
            epsilon=context.feasible_mapping_epsilon,
            bs_per_antenna_max_output_watt=(
                context.bs_per_antenna_max_output_watt
            ),
        )


def _normalized_features_to_channel(
    normalized_features: Tensor,
    mean: Tensor,
    standard_deviation: Tensor,
) -> Tensor:
    restored = normalized_features * standard_deviation + mean
    elements = restored.shape[-1] // 2
    per_user = torch.complex(
        restored[..., :elements], restored[..., elements:]
    )
    return per_user.transpose(-2, -1)


def _physical_equivalent_innovation_standard_deviation(
    normalized_standard_deviation: Tensor,
    normalizer_standard_deviation: Tensor,
) -> Tensor:
    """Return per-user RMS complex innovation scale in physical units."""
    if normalized_standard_deviation.ndim != 2:
        raise ValueError(
            "normalized standard deviation must have shape (batch, K)"
        )
    feature_size = normalizer_standard_deviation.numel()
    if feature_size == 0 or feature_size % 2 != 0:
        raise ValueError(
            "normalizer standard deviation must have shape (2N,)"
        )
    number_of_elements = feature_size // 2
    real_scale = normalizer_standard_deviation[:number_of_elements]
    imaginary_scale = normalizer_standard_deviation[number_of_elements:]
    physical_rms_per_normalized_unit = torch.sqrt(
        torch.mean(real_scale.square() + imaginary_scale.square())
    )
    return (
        normalized_standard_deviation
        * physical_rms_per_normalized_unit
    )


def _channel_to_normalized_features(
    channel: Tensor,
    mean: Tensor,
    standard_deviation: Tensor,
) -> Tensor:
    """Convert a physical `(batch, N, K)` channel to GRU output features."""
    if channel.ndim != 3 or not torch.is_complex(channel):
        raise ValueError("channel must be complex with shape (batch, N, K)")
    per_user = channel.transpose(-2, -1)
    features = torch.cat((per_user.real, per_user.imag), dim=-1)
    if mean.shape != (features.shape[-1],):
        raise ValueError("normalizer mean does not match channel elements")
    if standard_deviation.shape != (features.shape[-1],):
        raise ValueError(
            "normalizer standard deviation does not match channel elements"
        )
    return (features - mean) / standard_deviation


def _causal_channel_variation_from_normalized_history(
    normalized_history: Tensor,
    mean: Tensor,
    standard_deviation: Tensor,
    epsilon: float = 1.0e-30,
) -> Tensor:
    """Compute per-user variation from the observed physical CSI history."""
    if normalized_history.ndim != 4:
        raise ValueError(
            "normalized_history must have shape (batch, K, history, 2N)"
        )
    if normalized_history.shape[2] == 1:
        return torch.zeros(
            normalized_history.shape[:2],
            dtype=normalized_history.dtype,
            device=normalized_history.device,
        )
    if epsilon <= 0.0:
        raise ValueError("causal channel variation epsilon must be positive")
    restored = normalized_history * standard_deviation + mean
    elements = restored.shape[-1] // 2
    history = torch.complex(
        restored[..., :elements], restored[..., elements:]
    )
    increments = history[:, :, 1:, :] - history[:, :, :-1, :]
    numerator = torch.sum(torch.abs(increments) ** 2, dim=-1)
    denominator = torch.clamp(
        torch.sum(torch.abs(history[:, :, 1:, :]) ** 2, dim=-1),
        min=epsilon,
    )
    return torch.mean(numerator / denominator, dim=-1)


def _predictor_history_features(
    normalized_history: Tensor,
    input_mode: str,
) -> Tensor:
    if input_mode == "channel_only":
        return normalized_history
    if input_mode != "channel_and_first_difference":
        raise ValueError("predictor input mode is unsupported")
    first_difference = torch.zeros_like(normalized_history)
    first_difference[..., 1:, :] = (
        normalized_history[..., 1:, :]
        - normalized_history[..., :-1, :]
    )
    return torch.cat((normalized_history, first_difference), dim=-1)


def _scenario_user_sinr(
    *,
    bs_to_surface_channel: Tensor,
    scenario_channels: Tensor,
    controls: TorchControls,
    reflection_user_mask: Tensor,
    surface_noise_power_watt: float,
    receiver_noise_power_watt: Tensor,
) -> Tensor:
    """Evaluate `(batch, scenarios, N, K)` channels with shared controls."""
    if scenario_channels.ndim != 4 or not torch.is_complex(
        scenario_channels
    ):
        raise ValueError(
            "scenario_channels must be complex with shape (batch, S, N, K)"
        )
    expanded_controls = TorchControls(
        **{
            name: getattr(controls, name).unsqueeze(1)
            for name in controls.__dataclass_fields__
        }
    )
    return torch_user_sinr(
        bs_to_surface_channel=bs_to_surface_channel,
        surface_to_user_channel=scenario_channels,
        controls=expanded_controls,
        reflection_user_mask=reflection_user_mask,
        surface_noise_power_watt=surface_noise_power_watt,
        receiver_noise_power_watt=receiver_noise_power_watt,
    )


def _normalized_sinr_shortfall(sinr: Tensor, minimum_sinr: Tensor) -> Tensor:
    positive = minimum_sinr > 0.0
    denominator = torch.where(positive, minimum_sinr, torch.ones_like(minimum_sinr))
    return torch.where(
        positive,
        torch.relu((minimum_sinr - sinr) / denominator),
        torch.zeros_like(sinr),
    )


def _interpolate_design_sinr(
    payload_bits: Tensor,
    context: PADUContext,
) -> Tensor:
    payload_grid = context.design_payload_grid_bits.to(
        dtype=payload_bits.dtype,
        device=payload_bits.device,
    )
    sinr_grid = context.design_sinr_grid.to(
        dtype=payload_bits.dtype,
        device=payload_bits.device,
    )
    if payload_grid.ndim != 1 or payload_grid.numel() < 2:
        raise ValueError("design payload grid must contain at least two values")
    expected = (context.minimum_fbl_sinr.numel(), payload_grid.numel())
    if sinr_grid.shape != expected:
        raise ValueError("design SINR grid must have shape (K, G)")
    if not torch.all(payload_grid[1:] > payload_grid[:-1]):
        raise ValueError("design payload grid must be strictly increasing")
    clipped = torch.clamp(
        payload_bits,
        min=float(payload_grid[0]),
        max=float(payload_grid[-1]),
    )
    upper = torch.searchsorted(payload_grid, clipped, right=True)
    upper = torch.clamp(upper, min=1, max=payload_grid.numel() - 1)
    lower = upper - 1
    lower_payload = payload_grid[lower]
    upper_payload = payload_grid[upper]
    fraction = (clipped - lower_payload) / (upper_payload - lower_payload)
    expanded_grid = sinr_grid.unsqueeze(0).expand(payload_bits.shape[0], -1, -1)
    lower_sinr = torch.gather(expanded_grid, 2, lower.unsqueeze(-1)).squeeze(-1)
    upper_sinr = torch.gather(expanded_grid, 2, upper.unsqueeze(-1)).squeeze(-1)
    return lower_sinr + fraction * (upper_sinr - lower_sinr)


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(tensor)))
