"""Pre-experiment link budget and finite-blocklength diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .channels import umi_street_canyon_los_path_loss_db
from .config import ExperimentConfig
from .physics import finite_blocklength_spectral_efficiency
from .units import dbm_to_watt, noise_power_watt


@dataclass(frozen=True)
class LinkRange:
    name: str
    minimum_distance_3d_m: float
    maximum_distance_3d_m: float
    minimum_path_loss_db: float
    maximum_path_loss_db: float


@dataclass(frozen=True)
class ExperimentAudit:
    links: tuple[LinkRange, ...]
    receiver_noise_power_watt: float
    surface_noise_power_watt: float
    bs_power_budget_watt: float
    bs_per_antenna_power_budget_watt: tuple[float, ...] | None
    surface_total_power_budget_watt: float
    surface_per_element_power_budget_watt: float
    minimum_surface_noise_output_watt: float
    surface_total_noise_budget_margin_watt: float
    surface_per_element_noise_budget_margin_watt: float
    fbl_positive_rate_sinr_threshold_linear: float
    minimum_fbl_sinr_threshold_linear_per_user: tuple[float, ...]
    optimization_design_fbl_sinr_threshold_linear_per_user: tuple[float, ...]


@lru_cache(maxsize=None)
def finite_blocklength_positive_rate_sinr_threshold(
    blocklength: int,
    decoding_error_probability: float,
) -> float:
    return finite_blocklength_rate_sinr_threshold(
        blocklength,
        decoding_error_probability,
        0.0,
    )


@lru_cache(maxsize=None)
def finite_blocklength_rate_sinr_threshold(
    blocklength: int,
    decoding_error_probability: float,
    minimum_spectral_efficiency: float,
) -> float:
    if blocklength <= 0:
        raise ValueError("blocklength must be positive")
    if not 0.0 < decoding_error_probability < 0.5:
        raise ValueError(
            "decoding_error_probability must lie in (0, 0.5)"
        )
    if (
        not np.isfinite(minimum_spectral_efficiency)
        or minimum_spectral_efficiency < 0.0
    ):
        raise ValueError(
            "minimum_spectral_efficiency must be finite and non-negative"
        )

    def raw_rate(sinr: float) -> float:
        value = finite_blocklength_spectral_efficiency(
            np.array([sinr], dtype=np.float64),
            blocklength,
            decoding_error_probability,
        )[0]
        if value > 0.0:
            return float(value)
        dispersion = 2.0 * sinr / (1.0 + sinr)
        from scipy.stats import norm

        return float(
            np.log2(1.0 + sinr)
            - norm.isf(decoding_error_probability)
            / np.log(2.0)
            * np.sqrt(dispersion / blocklength)
        )

    lower = np.finfo(np.float64).eps

    def threshold_gap(sinr: float) -> float:
        return raw_rate(sinr) - minimum_spectral_efficiency

    if threshold_gap(lower) >= 0.0:
        raise RuntimeError(
            "failed to identify the negative-rate side of the FBL threshold"
        )
    upper = 1.0
    while threshold_gap(upper) <= 0.0:
        upper *= 2.0
        if not np.isfinite(upper):
            raise RuntimeError("failed to bracket the FBL rate threshold")
    return float(brentq(threshold_gap, lower, upper))


def audit_experiment(config: ExperimentConfig) -> ExperimentAudit:
    surface_noise = noise_power_watt(
        config.noise.noise_psd_dbm_per_hz,
        config.noise.bandwidth_hz,
        config.noise.star_noise_figure_db,
    )
    surface_total_budget = dbm_to_watt(
        config.power.star_total_max_output_dbm
    )
    surface_per_element_budget = dbm_to_watt(
        config.power.star_per_element_max_output_dbm
    )
    minimum_surface_noise_output = (
        config.array.star_elements * surface_noise
    )
    total_margin = surface_total_budget - minimum_surface_noise_output
    element_margin = surface_per_element_budget - surface_noise
    if total_margin < 0.0:
        raise ValueError(
            "active STAR-RIS total output-power budget is below the "
            "minimum amplified surface-noise output N * sigma_S^2"
        )
    if element_margin < 0.0:
        raise ValueError(
            "active STAR-RIS per-element output-power budget is below "
            "the minimum surface-noise output sigma_S^2"
        )

    links = [
        _fixed_link_range(
            "BS-to-STAR-RIS",
            config.geometry.bs_position_m,
            config.geometry.star_center_position_m,
            config.channel.carrier_frequency_ghz,
        ),
        _region_link_range(
            "STAR-RIS-to-reflection-region",
            config.geometry.star_center_position_m,
            config.geometry.user_height_m,
            config.geometry.reflection_region,
            config.channel.carrier_frequency_ghz,
        ),
        _region_link_range(
            "STAR-RIS-to-transmission-region",
            config.geometry.star_center_position_m,
            config.geometry.user_height_m,
            config.geometry.transmission_region,
            config.channel.carrier_frequency_ghz,
        ),
    ]
    per_antenna_power_dbm = config.power.bs_per_antenna_max_output_dbm
    per_antenna_power_watt = (
        None
        if per_antenna_power_dbm is None
        else dbm_to_watt(per_antenna_power_dbm)
    )
    return ExperimentAudit(
        links=tuple(links),
        receiver_noise_power_watt=noise_power_watt(
            config.noise.noise_psd_dbm_per_hz,
            config.noise.bandwidth_hz,
            config.noise.receiver_noise_figure_db,
        ),
        surface_noise_power_watt=surface_noise,
        bs_power_budget_watt=dbm_to_watt(
            config.power.bs_max_output_dbm
        ),
        bs_per_antenna_power_budget_watt=(
            None
            if per_antenna_power_watt is None
            else (per_antenna_power_watt,) * config.array.bs_antennas
        ),
        surface_total_power_budget_watt=surface_total_budget,
        surface_per_element_power_budget_watt=surface_per_element_budget,
        minimum_surface_noise_output_watt=minimum_surface_noise_output,
        surface_total_noise_budget_margin_watt=total_margin,
        surface_per_element_noise_budget_margin_watt=element_margin,
        fbl_positive_rate_sinr_threshold_linear=(
            finite_blocklength_positive_rate_sinr_threshold(
                config.finite_blocklength.blocklength,
                config.finite_blocklength.decoding_error_probability,
            )
        ),
        minimum_fbl_sinr_threshold_linear_per_user=tuple(
            0.0
            if minimum_rate == 0.0
            else finite_blocklength_rate_sinr_threshold(
                config.finite_blocklength.blocklength,
                config.finite_blocklength.decoding_error_probability,
                minimum_rate,
            )
            for minimum_rate in (
                config.finite_blocklength.minimum_spectral_efficiency_per_user
            )
        ),
        optimization_design_fbl_sinr_threshold_linear_per_user=tuple(
            0.0
            if minimum_rate == 0.0
            else finite_blocklength_rate_sinr_threshold(
                config.finite_blocklength.blocklength,
                config.finite_blocklength.decoding_error_probability,
                minimum_rate,
            )
            for minimum_rate in (
                config.finite_blocklength
                .optimization_design_spectral_efficiency_per_user
            )
        ),
    )


def _fixed_link_range(
    name: str,
    first_position_m: np.ndarray,
    second_position_m: np.ndarray,
    carrier_frequency_ghz: float,
) -> LinkRange:
    distance = float(
        np.linalg.norm(first_position_m - second_position_m)
    )
    loss = float(
        umi_street_canyon_los_path_loss_db(
            distance,
            carrier_frequency_ghz,
        )
    )
    return LinkRange(name, distance, distance, loss, loss)


def _region_link_range(
    name: str,
    surface_position_m: np.ndarray,
    user_height_m: float,
    region,
    carrier_frequency_ghz: float,
) -> LinkRange:
    corners = np.array(
        [
            [region.x_min, region.y_min, user_height_m],
            [region.x_min, region.y_max, user_height_m],
            [region.x_max, region.y_min, user_height_m],
            [region.x_max, region.y_max, user_height_m],
        ],
        dtype=np.float64,
    )
    nearest_xy = np.array(
        [
            np.clip(
                surface_position_m[0],
                region.x_min,
                region.x_max,
            ),
            np.clip(
                surface_position_m[1],
                region.y_min,
                region.y_max,
            ),
            user_height_m,
        ],
        dtype=np.float64,
    )
    minimum_distance = float(
        np.linalg.norm(nearest_xy - surface_position_m)
    )
    maximum_distance = float(
        np.max(
            np.linalg.norm(
                corners - surface_position_m,
                axis=1,
            )
        )
    )
    minimum_loss = float(
        umi_street_canyon_los_path_loss_db(
            minimum_distance,
            carrier_frequency_ghz,
        )
    )
    maximum_loss = float(
        umi_street_canyon_los_path_loss_db(
            maximum_distance,
            carrier_frequency_ghz,
        )
    )
    return LinkRange(
        name=name,
        minimum_distance_3d_m=minimum_distance,
        maximum_distance_3d_m=maximum_distance,
        minimum_path_loss_db=minimum_loss,
        maximum_path_loss_db=maximum_loss,
    )
