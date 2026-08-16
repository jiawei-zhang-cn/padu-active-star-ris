"""Differentiable warm-start mapping and physical objective in PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TorchControls:
    beamforming: Tensor
    reflection_coefficients: Tensor
    transmission_coefficients: Tensor
    power_gain: Tensor
    reflection_power_split: Tensor
    reflection_phase_rad: Tensor
    transmission_phase_rad: Tensor
    beam_scaling: Tensor


def decode_ideal_passive_controls(
    raw_output: Tensor,
    number_of_bs_antennas: int,
    number_of_surface_elements: int,
    number_of_users: int,
    bs_max_output_watt: float,
    epsilon: float,
    bs_per_antenna_max_output_watt: Tensor | None = None,
) -> TorchControls:
    expected = (
        2 * number_of_bs_antennas * number_of_users
        + 3 * number_of_surface_elements
    )
    if raw_output.shape[-1] != expected:
        raise ValueError("raw_output has an invalid last dimension")
    if bs_max_output_watt <= 0.0 or epsilon <= 0.0:
        raise ValueError("power budget and epsilon must be positive")

    batch_shape = raw_output.shape[:-1]
    beam_values = number_of_bs_antennas * number_of_users
    offset = 0
    raw_beam_real = raw_output[..., offset : offset + beam_values]
    offset += beam_values
    raw_beam_imaginary = raw_output[..., offset : offset + beam_values]
    offset += beam_values
    raw_split = raw_output[..., offset : offset + number_of_surface_elements]
    offset += number_of_surface_elements
    raw_phase_r = raw_output[..., offset : offset + number_of_surface_elements]
    offset += number_of_surface_elements
    raw_phase_t = raw_output[..., offset : offset + number_of_surface_elements]

    raw_beam = torch.complex(
        raw_beam_real,
        raw_beam_imaginary,
    ).reshape(
        *batch_shape,
        number_of_users,
        number_of_bs_antennas,
    ).transpose(-2, -1)
    beam_scaling = _bs_beam_scaling(
        raw_beam,
        bs_max_output_watt,
        bs_per_antenna_max_output_watt,
        epsilon,
    )
    beamforming = raw_beam * beam_scaling[..., None, None]

    reflection_split = torch.sigmoid(raw_split)
    reflection_amplitude, transmission_amplitude = (
        _stable_energy_splitting_amplitudes(raw_split)
    )
    phase_r = 2.0 * math.pi * torch.sigmoid(raw_phase_r)
    phase_t = 2.0 * math.pi * torch.sigmoid(raw_phase_t)
    power_gain = torch.ones_like(reflection_split)
    reflection = reflection_amplitude * torch.exp(1j * phase_r)
    transmission = transmission_amplitude * torch.exp(
        1j * phase_t
    )
    return TorchControls(
        beamforming=beamforming,
        reflection_coefficients=reflection,
        transmission_coefficients=transmission,
        power_gain=power_gain,
        reflection_power_split=reflection_split,
        reflection_phase_rad=phase_r,
        transmission_phase_rad=phase_t,
        beam_scaling=beam_scaling,
    )


def decode_feasible_controls(
    raw_output: Tensor,
    bs_to_surface_channel: Tensor,
    number_of_bs_antennas: int,
    number_of_surface_elements: int,
    number_of_users: int,
    bs_max_output_watt: float,
    star_total_max_output_watt: float,
    star_per_element_max_output_watt: float,
    star_max_power_gain: float,
    surface_noise_power_watt: float,
    epsilon: float,
    bs_per_antenna_max_output_watt: Tensor | None = None,
) -> TorchControls:
    if raw_output.shape[-1] != (
        2 * number_of_bs_antennas * number_of_users
        + 4 * number_of_surface_elements
    ):
        raise ValueError("raw_output has an invalid last dimension")
    if bs_to_surface_channel.shape != (
        number_of_surface_elements,
        number_of_bs_antennas,
    ):
        raise ValueError("bs_to_surface_channel must have shape (N, M)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if star_max_power_gain <= 1.0:
        raise ValueError("star_max_power_gain must exceed one")
    if surface_noise_power_watt <= 0.0:
        raise ValueError("surface_noise_power_watt must be positive")
    if (
        star_per_element_max_output_watt < surface_noise_power_watt
        or star_total_max_output_watt
        < number_of_surface_elements * surface_noise_power_watt
    ):
        raise ValueError("active STAR-RIS budgets fail the unit-gain noise floor")

    batch_shape = raw_output.shape[:-1]
    beam_values = number_of_bs_antennas * number_of_users
    offset = 0
    raw_beam_real = raw_output[..., offset : offset + beam_values]
    offset += beam_values
    raw_beam_imaginary = raw_output[..., offset : offset + beam_values]
    offset += beam_values
    raw_gain = raw_output[..., offset : offset + number_of_surface_elements]
    offset += number_of_surface_elements
    raw_split = raw_output[..., offset : offset + number_of_surface_elements]
    offset += number_of_surface_elements
    raw_phase_r = raw_output[..., offset : offset + number_of_surface_elements]
    offset += number_of_surface_elements
    raw_phase_t = raw_output[..., offset : offset + number_of_surface_elements]

    raw_beam = torch.complex(
        raw_beam_real,
        raw_beam_imaginary,
    ).reshape(
        *batch_shape,
        number_of_users,
        number_of_bs_antennas,
    ).transpose(-2, -1)

    gain_ceiling = min(
        star_max_power_gain,
        star_per_element_max_output_watt / surface_noise_power_watt,
        star_total_max_output_watt
        / (number_of_surface_elements * surface_noise_power_watt),
    )
    power_gain = 1.0 + (gain_ceiling - 1.0) * torch.sigmoid(raw_gain)
    reflection_split = torch.sigmoid(raw_split)
    reflection_amplitude, transmission_amplitude = (
        _stable_energy_splitting_amplitudes(raw_split)
    )
    phase_r = 2.0 * math.pi * torch.sigmoid(raw_phase_r)
    phase_t = 2.0 * math.pi * torch.sigmoid(raw_phase_t)

    incident = torch.einsum("nm,...mk->...nk", bs_to_surface_channel, raw_beam)
    incident_power = torch.sum(torch.abs(incident) ** 2, dim=-1)
    bs_limit = _bs_beam_scaling(
        raw_beam,
        bs_max_output_watt,
        bs_per_antenna_max_output_watt,
        epsilon,
    )
    per_element_numerator = (
        star_per_element_max_output_watt
        - surface_noise_power_watt * power_gain
    )
    per_element_limit = torch.sqrt(
        torch.clamp(per_element_numerator, min=0.0)
        / (power_gain * incident_power + epsilon)
    ).amin(dim=-1)
    total_numerator = (
        star_total_max_output_watt
        - surface_noise_power_watt * torch.sum(power_gain, dim=-1)
    )
    total_limit = torch.sqrt(
        torch.clamp(total_numerator, min=0.0)
        / (
            torch.sum(power_gain * incident_power, dim=-1)
            + epsilon
        )
    )
    beam_scaling = torch.minimum(
        torch.ones_like(bs_limit),
        torch.minimum(bs_limit, torch.minimum(per_element_limit, total_limit)),
    )
    beamforming = raw_beam * beam_scaling[..., None, None]

    gain_amplitude = torch.sqrt(power_gain)
    reflection = gain_amplitude * reflection_amplitude * torch.exp(
        1j * phase_r
    )
    transmission = gain_amplitude * transmission_amplitude * torch.exp(
        1j * phase_t
    )
    return TorchControls(
        beamforming=beamforming,
        reflection_coefficients=reflection,
        transmission_coefficients=transmission,
        power_gain=power_gain,
        reflection_power_split=reflection_split,
        reflection_phase_rad=phase_r,
        transmission_phase_rad=phase_t,
        beam_scaling=beam_scaling,
    )


def _bs_beam_scaling(
    raw_beam: Tensor,
    bs_max_output_watt: float,
    bs_per_antenna_max_output_watt: Tensor | None,
    epsilon: float,
) -> Tensor:
    if bs_max_output_watt <= 0.0 or epsilon <= 0.0:
        raise ValueError("power budget and epsilon must be positive")
    raw_bs_power = torch.sum(torch.abs(raw_beam) ** 2, dim=(-2, -1))
    total_limit = torch.sqrt(
        torch.as_tensor(
            bs_max_output_watt,
            dtype=raw_bs_power.dtype,
            device=raw_bs_power.device,
        )
        / (raw_bs_power + epsilon)
    )
    scaling = torch.minimum(torch.ones_like(total_limit), total_limit)
    if bs_per_antenna_max_output_watt is None:
        return scaling
    per_antenna = torch.as_tensor(
        bs_per_antenna_max_output_watt,
        dtype=raw_bs_power.dtype,
        device=raw_bs_power.device,
    )
    antennas = raw_beam.shape[-2]
    if per_antenna.shape != (antennas,):
        raise ValueError(
            "bs_per_antenna_max_output_watt must have shape (M,)"
        )
    if not torch.all(torch.isfinite(per_antenna)) or torch.any(
        per_antenna <= 0.0
    ):
        raise ValueError(
            "BS per-antenna power budgets must be finite and positive"
        )
    antenna_power = torch.sum(torch.abs(raw_beam) ** 2, dim=-1)
    antenna_limit = torch.sqrt(
        per_antenna / (antenna_power + epsilon)
    ).amin(dim=-1)
    return torch.minimum(scaling, antenna_limit)


def _stable_energy_splitting_amplitudes(
    raw_split: Tensor,
) -> tuple[Tensor, Tensor]:
    reflection_amplitude = torch.exp(0.5 * F.logsigmoid(raw_split))
    transmission_amplitude = torch.exp(0.5 * F.logsigmoid(-raw_split))
    return reflection_amplitude, transmission_amplitude


def torch_user_sinr(
    bs_to_surface_channel: Tensor,
    surface_to_user_channel: Tensor,
    controls: TorchControls,
    reflection_user_mask: Tensor,
    surface_noise_power_watt: float,
    receiver_noise_power_watt: Tensor,
) -> Tensor:
    """Accept H with shape (..., N, K) and controls with matching batch axes."""
    if not torch.is_complex(bs_to_surface_channel):
        raise ValueError("bs_to_surface_channel must be complex")
    if not torch.is_complex(surface_to_user_channel):
        raise ValueError("surface_to_user_channel must be complex")
    if surface_to_user_channel.shape[-2] != bs_to_surface_channel.shape[0]:
        raise ValueError("surface element dimensions do not match")
    number_of_users = surface_to_user_channel.shape[-1]
    if reflection_user_mask.shape != (number_of_users,):
        raise ValueError("reflection_user_mask must have shape (K,)")

    coefficient_batch_dimensions = controls.reflection_coefficients.ndim - 1
    mask_shape = (
        (1,) * coefficient_batch_dimensions
        + (number_of_users, 1)
    )
    coefficient = torch.where(
        reflection_user_mask.reshape(mask_shape),
        controls.reflection_coefficients.unsqueeze(-2),
        controls.transmission_coefficients.unsqueeze(-2),
    )
    conjugate_user_rows = torch.conj(
        surface_to_user_channel.transpose(-2, -1)
    )
    effective = (conjugate_user_rows * coefficient) @ bs_to_surface_channel
    amplitudes = torch.einsum(
        "...km,...mj->...kj",
        effective,
        controls.beamforming,
    )
    powers = torch.abs(amplitudes) ** 2
    desired = torch.diagonal(powers, dim1=-2, dim2=-1)
    interference = torch.sum(powers, dim=-1) - desired
    surface_noise = surface_noise_power_watt * torch.sum(
        torch.abs(conjugate_user_rows) ** 2
        * torch.abs(coefficient) ** 2,
        dim=-1,
    )
    return desired / (interference + surface_noise + receiver_noise_power_watt)


def torch_fbl_spectral_efficiency(
    sinr: Tensor,
    blocklength: int,
    decoding_error_probability: float,
) -> Tensor:
    if blocklength <= 0:
        raise ValueError("blocklength must be positive")
    if not 0.0 < decoding_error_probability < 0.5:
        raise ValueError("decoding_error_probability must lie in (0, 0.5)")
    error = torch.tensor(
        decoding_error_probability,
        dtype=sinr.dtype,
        device=sinr.device,
    )
    inverse_q = math.sqrt(2.0) * torch.special.erfinv(1.0 - 2.0 * error)
    dispersion = 2.0 * sinr / (1.0 + sinr)
    positive_dispersion = dispersion > 0.0
    safe_dispersion = torch.where(
        positive_dispersion,
        dispersion,
        torch.ones_like(dispersion),
    )
    dispersion_penalty = torch.where(
        positive_dispersion,
        torch.sqrt(safe_dispersion / blocklength),
        torch.zeros_like(dispersion),
    )
    raw = (
        torch.log1p(sinr) / math.log(2.0)
        - inverse_q
        / math.log(2.0)
        * dispersion_penalty
    )
    return torch.clamp(raw, min=0.0)


def sum_fbl_throughput(
    bs_to_surface_channel: Tensor,
    surface_to_user_channel: Tensor,
    controls: TorchControls,
    reflection_user_mask: Tensor,
    surface_noise_power_watt: float,
    receiver_noise_power_watt: Tensor,
    blocklength: int,
    decoding_error_probability: float,
    bandwidth_hz: float,
) -> Tensor:
    sinr = torch_user_sinr(
        bs_to_surface_channel,
        surface_to_user_channel,
        controls,
        reflection_user_mask,
        surface_noise_power_watt,
        receiver_noise_power_watt,
    )
    spectral_efficiency = torch_fbl_spectral_efficiency(
        sinr,
        blocklength,
        decoding_error_probability,
    )
    throughput = (
        (1.0 - decoding_error_probability)
        * bandwidth_hz
        * spectral_efficiency
    )
    return torch.sum(throughput, dim=-1)


def surface_output_powers(
    bs_to_surface_channel: Tensor,
    controls: TorchControls,
    surface_noise_power_watt: float,
) -> Tensor:
    incident = torch.einsum(
        "nm,...mk->...nk",
        bs_to_surface_channel,
        controls.beamforming,
    )
    return controls.power_gain * (
        torch.sum(torch.abs(incident) ** 2, dim=-1)
        + surface_noise_power_watt
    )
