"""Fixed deployment and mobile active STAR-RIS channel generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .arrays import (
    bs_to_surface_los_matrix,
    centered_ula_offsets,
    centered_ura_offsets,
    surface_to_user_los_vector,
    unit_vector,
)
from .channels import (
    complex_standard_normal,
    gauss_markov_step,
    jakes_correlation,
    rician_channel,
    umi_street_canyon_los_path_gain,
)
from .config import ExperimentConfig
from .mobility import (
    WaypointState,
    advance_random_waypoint,
    initialize_random_waypoint,
)
from .units import wavelength_m


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FixedDeployment:
    bs_to_surface_channel: ComplexArray
    bs_offsets_m: FloatArray
    surface_offsets_m: FloatArray
    wavelength_m: float


@dataclass(frozen=True)
class MobileTrajectory:
    positions_m: FloatArray
    surface_to_user_channels: ComplexArray
    user_nlos_components: ComplexArray
    user_speeds_m_per_s: tuple[float, ...]
    user_rician_factor_db: tuple[float, ...]


def create_fixed_deployment(
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> FixedDeployment:
    wavelength = wavelength_m(config.channel.carrier_frequency_ghz)
    array = config.array
    geometry = config.geometry
    bs_offsets = centered_ula_offsets(
        array.bs_antennas,
        array.bs_spacing_wavelengths * wavelength,
        array.bs_ula_axis,
    )
    surface_offsets = centered_ura_offsets(
        array.star_rows,
        array.star_columns,
        array.star_horizontal_spacing_wavelengths * wavelength,
        array.star_vertical_spacing_wavelengths * wavelength,
        array.star_horizontal_axis,
        array.star_vertical_axis,
    )
    displacement = (
        geometry.star_center_position_m - geometry.bs_position_m
    )
    distance = float(np.linalg.norm(displacement))
    direction = unit_vector(displacement, "BS-to-surface displacement")
    los = bs_to_surface_los_matrix(
        bs_offsets,
        surface_offsets,
        direction,
        distance,
        wavelength,
        config.channel.center_path_phase_mode,
    )
    nlos = complex_standard_normal(
        (array.star_elements, array.bs_antennas),
        rng,
    )
    gain = float(
        umi_street_canyon_los_path_gain(
            distance,
            config.channel.carrier_frequency_ghz,
        )
    )
    channel = rician_channel(
        gain,
        config.channel.bs_star_rician_factor_db,
        los,
        nlos,
    )
    return FixedDeployment(
        bs_to_surface_channel=channel,
        bs_offsets_m=bs_offsets,
        surface_offsets_m=surface_offsets,
        wavelength_m=wavelength,
    )


def generate_mobile_trajectory(
    config: ExperimentConfig,
    deployment: FixedDeployment,
    number_of_slots: int,
    rng: np.random.Generator,
    user_speeds_m_per_s: tuple[float, ...] | None = None,
    user_rician_factor_db: tuple[float, ...] | None = None,
) -> MobileTrajectory:
    if number_of_slots <= 0:
        raise ValueError("number_of_slots must be positive")
    speeds = (
        config.mobility.user_speeds_m_per_s
        if user_speeds_m_per_s is None
        else tuple(float(value) for value in user_speeds_m_per_s)
    )
    factors = (
        config.channel.user_rician_factor_db
        if user_rician_factor_db is None
        else tuple(float(value) for value in user_rician_factor_db)
    )
    users = config.number_of_users
    if len(speeds) != users or len(factors) != users:
        raise ValueError("speed and Rician-factor tuples must contain K values")
    if any(not np.isfinite(value) or value < 0.0 for value in speeds):
        raise ValueError("user speeds must be finite and non-negative")
    if any(not np.isfinite(value) for value in factors):
        raise ValueError("user Rician factors must be finite")

    elements = config.array.star_elements
    positions = np.empty((number_of_slots + 1, users, 3), dtype=np.float64)
    channels = np.empty(
        (number_of_slots + 1, elements, users),
        dtype=np.complex64,
    )
    nlos_history = np.empty_like(channels)
    states: list[WaypointState] = []
    nlos_components: list[ComplexArray] = []

    for user, side in enumerate(config.mobility.user_sides):
        region = _region_for_side(config, side)
        states.append(
            initialize_random_waypoint(
                region,
                speeds[user],
                config.mobility.arrival_tolerance_m,
                rng,
            )
        )
        nlos_components.append(complex_standard_normal((elements,), rng))

    for time_index in range(number_of_slots + 1):
        for user, side in enumerate(config.mobility.user_sides):
            if time_index > 0:
                region = _region_for_side(config, side)
                states[user] = advance_random_waypoint(
                    states[user],
                    config.channel.time_slot_s,
                    config.mobility.arrival_tolerance_m,
                    lambda region=region: region.sample(rng),
                )
                correlation = jakes_correlation(
                    speeds[user],
                    config.channel.carrier_frequency_ghz,
                    config.channel.time_slot_s,
                )
                nlos_components[user] = gauss_markov_step(
                    nlos_components[user],
                    correlation,
                    complex_standard_normal((elements,), rng),
                )

            position = np.array(
                [
                    states[user].position_xy_m[0],
                    states[user].position_xy_m[1],
                    config.geometry.user_height_m,
                ],
                dtype=np.float64,
            )
            positions[time_index, user] = position
            nlos_history[time_index, :, user] = nlos_components[user]
            channels[time_index, :, user] = _surface_to_user_channel(
                config,
                deployment,
                position,
                factors[user],
                nlos_components[user],
            )

    return MobileTrajectory(
        positions_m=positions,
        surface_to_user_channels=channels,
        user_nlos_components=nlos_history,
        user_speeds_m_per_s=speeds,
        user_rician_factor_db=factors,
    )


def _region_for_side(config: ExperimentConfig, side: str):
    if side == "R":
        return config.geometry.reflection_region
    if side == "T":
        return config.geometry.transmission_region
    raise ValueError("side must be 'R' or 'T'")


def _surface_to_user_channel(
    config: ExperimentConfig,
    deployment: FixedDeployment,
    user_position_m: FloatArray,
    rician_factor_db: float,
    nlos_component: ComplexArray,
) -> ComplexArray:
    displacement = user_position_m - config.geometry.star_center_position_m
    distance = float(np.linalg.norm(displacement))
    direction = unit_vector(displacement, "surface-to-user displacement")
    los = surface_to_user_los_vector(
        deployment.surface_offsets_m,
        direction,
        distance,
        deployment.wavelength_m,
        config.channel.center_path_phase_mode,
    )
    gain = float(
        umi_street_canyon_los_path_gain(
            distance,
            config.channel.carrier_frequency_ghz,
        )
    )
    return rician_channel(
        gain,
        rician_factor_db,
        los,
        nlos_component,
    )

