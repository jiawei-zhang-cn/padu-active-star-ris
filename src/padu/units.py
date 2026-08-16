"""Unit conversions used by the physical simulation."""

from __future__ import annotations

import math


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 10.0)


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        raise ValueError("value must be positive")
    return 10.0 * math.log10(float(value))


def dbm_to_watt(value_dbm: float) -> float:
    return 10.0 ** ((float(value_dbm) - 30.0) / 10.0)


def watt_to_dbm(value_watt: float) -> float:
    if value_watt <= 0.0:
        raise ValueError("value_watt must be positive")
    return 10.0 * math.log10(float(value_watt)) + 30.0


def ghz_to_hz(value_ghz: float) -> float:
    if value_ghz <= 0.0:
        raise ValueError("value_ghz must be positive")
    return float(value_ghz) * 1.0e9


def wavelength_m(carrier_frequency_ghz: float) -> float:
    return SPEED_OF_LIGHT_M_PER_S / ghz_to_hz(carrier_frequency_ghz)


def noise_power_watt(
    noise_psd_dbm_per_hz: float,
    bandwidth_hz: float,
    noise_figure_db: float,
) -> float:
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive")
    return dbm_to_watt(
        float(noise_psd_dbm_per_hz)
        + 10.0 * math.log10(float(bandwidth_hz))
        + float(noise_figure_db)
    )


def integer_channel_uses(
    bandwidth_hz: float,
    packet_duration_s: float,
    absolute_tolerance: float = 1.0e-9,
) -> int:
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive")
    if packet_duration_s <= 0.0:
        raise ValueError("packet_duration_s must be positive")
    raw = float(bandwidth_hz) * float(packet_duration_s)
    rounded = round(raw)
    if rounded <= 0 or not math.isclose(
        raw,
        rounded,
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(
            "bandwidth_hz * packet_duration_s must be a positive integer"
        )
    return int(rounded)

