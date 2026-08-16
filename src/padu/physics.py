"""Active STAR-RIS power, signal, SINR, and short-packet metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


def active_star_coefficients(
    power_gain: FloatArray,
    reflection_power_split: FloatArray,
    reflection_phase_rad: FloatArray,
    transmission_phase_rad: FloatArray,
) -> tuple[ComplexArray, ComplexArray]:
    gain = np.asarray(power_gain, dtype=np.float64)
    split = np.asarray(reflection_power_split, dtype=np.float64)
    phase_r = np.asarray(reflection_phase_rad, dtype=np.float64)
    phase_t = np.asarray(transmission_phase_rad, dtype=np.float64)
    if not (gain.shape == split.shape == phase_r.shape == phase_t.shape):
        raise ValueError("all active STAR-RIS vectors must have the same shape")
    if gain.ndim != 1 or gain.size == 0:
        raise ValueError("active STAR-RIS vectors must be non-empty and one-dimensional")
    if np.any(gain < 1.0) or np.any(~np.isfinite(gain)):
        raise ValueError("power_gain must be finite and at least one")
    if np.any((split < 0.0) | (split > 1.0)) or np.any(~np.isfinite(split)):
        raise ValueError("reflection_power_split must lie in [0, 1]")
    if np.any(~np.isfinite(phase_r)) or np.any(~np.isfinite(phase_t)):
        raise ValueError("phase vectors must be finite")

    reflection = np.sqrt(gain * split) * np.exp(1j * phase_r)
    transmission = np.sqrt(gain * (1.0 - split)) * np.exp(1j * phase_t)
    return reflection.astype(np.complex64), transmission.astype(np.complex64)


def per_element_surface_output_power(
    bs_to_surface_channel: ComplexArray,
    beamforming: ComplexArray,
    power_gain: FloatArray,
    surface_noise_power_watt: float,
) -> FloatArray:
    channel = np.asarray(bs_to_surface_channel, dtype=np.complex64)
    beams = np.asarray(beamforming, dtype=np.complex64)
    gain = np.asarray(power_gain, dtype=np.float64)
    if channel.ndim != 2 or beams.ndim != 2:
        raise ValueError("channel and beamforming must be matrices")
    if channel.shape[1] != beams.shape[0]:
        raise ValueError("channel and beamforming dimensions do not align")
    if gain.shape != (channel.shape[0],):
        raise ValueError("power_gain length must equal the number of surface elements")
    if np.any(gain < 1.0):
        raise ValueError("power_gain must be at least one")
    if surface_noise_power_watt < 0.0:
        raise ValueError("surface_noise_power_watt must be non-negative")
    incident = channel @ beams
    incident_power = np.sum(np.abs(incident) ** 2, axis=1)
    return gain * (incident_power + surface_noise_power_watt)


def user_sinr(
    bs_to_surface_channel: ComplexArray,
    surface_to_user_channels: ComplexArray,
    beamforming: ComplexArray,
    reflection_coefficients: ComplexArray,
    transmission_coefficients: ComplexArray,
    user_sides: tuple[str, ...] | list[str],
    surface_noise_power_watt: float,
    receiver_noise_power_watt: float | FloatArray,
) -> FloatArray:
    channel_bs = np.asarray(bs_to_surface_channel, dtype=np.complex64)
    channel_users = np.asarray(surface_to_user_channels, dtype=np.complex64)
    beams = np.asarray(beamforming, dtype=np.complex64)
    coefficient_r = np.asarray(reflection_coefficients, dtype=np.complex64)
    coefficient_t = np.asarray(transmission_coefficients, dtype=np.complex64)

    if channel_bs.ndim != 2:
        raise ValueError("bs_to_surface_channel must have shape (N, M)")
    number_of_elements, number_of_bs_antennas = channel_bs.shape
    if channel_users.ndim != 2 or channel_users.shape[0] != number_of_elements:
        raise ValueError("surface_to_user_channels must have shape (N, K)")
    number_of_users = channel_users.shape[1]
    if beams.shape != (number_of_bs_antennas, number_of_users):
        raise ValueError("beamforming must have shape (M, K)")
    if coefficient_r.shape != (number_of_elements,) or coefficient_t.shape != (
        number_of_elements,
    ):
        raise ValueError("surface coefficient vectors must have length N")
    if len(user_sides) != number_of_users:
        raise ValueError("user_sides must contain one entry per user")
    if surface_noise_power_watt < 0.0:
        raise ValueError("surface_noise_power_watt must be non-negative")

    receiver_noise = np.asarray(receiver_noise_power_watt, dtype=np.float64)
    if receiver_noise.ndim == 0:
        receiver_noise = np.full(number_of_users, float(receiver_noise))
    if receiver_noise.shape != (number_of_users,) or np.any(receiver_noise <= 0.0):
        raise ValueError("receiver noise must be positive for every user")

    sinr = np.empty(number_of_users, dtype=np.float64)
    for user in range(number_of_users):
        side = user_sides[user]
        if side == "R":
            coefficient = coefficient_r
        elif side == "T":
            coefficient = coefficient_t
        else:
            raise ValueError("each user side must be 'R' or 'T'")

        h = channel_users[:, user]
        effective_row = (np.conj(h) * coefficient) @ channel_bs
        stream_amplitudes = effective_row @ beams
        stream_powers = np.abs(stream_amplitudes) ** 2
        desired = float(stream_powers[user])
        interference = float(np.sum(stream_powers) - desired)
        amplified_surface_noise = float(
            surface_noise_power_watt
            * np.sum(np.abs(h) ** 2 * np.abs(coefficient) ** 2)
        )
        denominator = (
            interference
            + amplified_surface_noise
            + float(receiver_noise[user])
        )
        sinr[user] = desired / denominator
    return sinr


def finite_blocklength_dispersion(sinr: FloatArray) -> FloatArray:
    values = np.asarray(sinr, dtype=np.float64)
    if np.any(values < 0.0) or np.any(~np.isfinite(values)):
        raise ValueError("sinr must be finite and non-negative")
    return 2.0 * values / (1.0 + values)


def finite_blocklength_spectral_efficiency(
    sinr: FloatArray,
    blocklength: int,
    decoding_error_probability: float,
) -> FloatArray:
    values = np.asarray(sinr, dtype=np.float64)
    if np.any(values < 0.0) or np.any(~np.isfinite(values)):
        raise ValueError("sinr must be finite and non-negative")
    if blocklength <= 0:
        raise ValueError("blocklength must be positive")
    if not 0.0 < decoding_error_probability < 0.5:
        raise ValueError("decoding_error_probability must lie in (0, 0.5)")

    dispersion = finite_blocklength_dispersion(values)
    penalty = (
        norm.isf(decoding_error_probability)
        / math.log(2.0)
        * np.sqrt(dispersion / blocklength)
    )
    raw = np.log2(1.0 + values) - penalty
    return np.maximum(raw, 0.0)


def finite_blocklength_throughput_bps(
    spectral_efficiency: FloatArray,
    bandwidth_hz: float,
    decoding_error_probability: float,
) -> FloatArray:
    rate = np.asarray(spectral_efficiency, dtype=np.float64)
    if np.any(rate < 0.0) or np.any(~np.isfinite(rate)):
        raise ValueError("spectral_efficiency must be finite and non-negative")
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive")
    if not 0.0 < decoding_error_probability < 0.5:
        raise ValueError("decoding_error_probability must lie in (0, 0.5)")
    return (
        (1.0 - decoding_error_probability)
        * float(bandwidth_hz)
        * rate
    )

