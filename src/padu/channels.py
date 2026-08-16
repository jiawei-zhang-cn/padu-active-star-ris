"""Path loss, Rician fading, and time-correlated mobile channels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

from .units import SPEED_OF_LIGHT_M_PER_S, db_to_linear, ghz_to_hz


ComplexArray = NDArray[np.complex64]


def umi_street_canyon_los_path_loss_db(
    distance_3d_m: float | NDArray[np.float64],
    carrier_frequency_ghz: float,
) -> NDArray[np.float64]:
    """Implement the LoS expression stated in the current paper."""
    distance = np.asarray(distance_3d_m, dtype=np.float64)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("distance_3d_m must be finite and positive")
    if carrier_frequency_ghz <= 0.0:
        raise ValueError("carrier_frequency_ghz must be positive")
    return (
        32.4
        + 21.0 * np.log10(distance)
        + 20.0 * np.log10(float(carrier_frequency_ghz))
    )


def umi_street_canyon_los_path_gain(
    distance_3d_m: float | NDArray[np.float64],
    carrier_frequency_ghz: float,
) -> NDArray[np.float64]:
    loss_db = umi_street_canyon_los_path_loss_db(
        distance_3d_m,
        carrier_frequency_ghz,
    )
    return np.power(10.0, -loss_db / 10.0)


def complex_standard_normal(
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> ComplexArray:
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("all shape dimensions must be positive")
    real = rng.standard_normal(shape)
    imaginary = rng.standard_normal(shape)
    return ((real + 1j * imaginary) / np.sqrt(2.0)).astype(np.complex64)


def rician_channel(
    path_gain: float,
    rician_factor_db: float,
    los_component: ComplexArray,
    nlos_component: ComplexArray,
) -> ComplexArray:
    if not np.isfinite(path_gain) or path_gain <= 0.0:
        raise ValueError("path_gain must be finite and positive")
    los = np.asarray(los_component, dtype=np.complex64)
    nlos = np.asarray(nlos_component, dtype=np.complex64)
    if los.shape != nlos.shape:
        raise ValueError("LoS and NLoS components must have the same shape")
    if not np.all(np.isfinite(los)) or not np.all(np.isfinite(nlos)):
        raise ValueError("channel components must be finite")
    factor = db_to_linear(rician_factor_db)
    channel = np.sqrt(path_gain) * (
        np.sqrt(factor / (factor + 1.0)) * los
        + np.sqrt(1.0 / (factor + 1.0)) * nlos
    )
    return channel.astype(np.complex64)


def doppler_frequency_hz(
    speed_m_per_s: float,
    carrier_frequency_ghz: float,
) -> float:
    if speed_m_per_s < 0.0 or not np.isfinite(speed_m_per_s):
        raise ValueError("speed_m_per_s must be finite and non-negative")
    return (
        float(speed_m_per_s)
        * ghz_to_hz(carrier_frequency_ghz)
        / SPEED_OF_LIGHT_M_PER_S
    )


def jakes_correlation(
    speed_m_per_s: float,
    carrier_frequency_ghz: float,
    slot_duration_s: float,
) -> float:
    if slot_duration_s <= 0.0 or not np.isfinite(slot_duration_s):
        raise ValueError("slot_duration_s must be finite and positive")
    doppler = doppler_frequency_hz(
        speed_m_per_s,
        carrier_frequency_ghz,
    )
    return float(j0(2.0 * np.pi * doppler * slot_duration_s))


def gauss_markov_step(
    previous_nlos: ComplexArray,
    correlation: float,
    innovation: ComplexArray,
) -> ComplexArray:
    previous = np.asarray(previous_nlos, dtype=np.complex64)
    new_noise = np.asarray(innovation, dtype=np.complex64)
    if previous.shape != new_noise.shape:
        raise ValueError("previous_nlos and innovation must have the same shape")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    updated = (
        correlation * previous
        + np.sqrt(max(0.0, 1.0 - correlation**2)) * new_noise
    )
    return updated.astype(np.complex64)

