"""Three-dimensional ULA and URA coordinate and response utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex64]


def unit_vector(value: FloatArray, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def centered_ula_offsets(
    number_of_antennas: int,
    spacing_m: float,
    axis: FloatArray,
) -> FloatArray:
    if number_of_antennas <= 0:
        raise ValueError("number_of_antennas must be positive")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    axis_unit = unit_vector(axis, "axis")
    indices = np.arange(number_of_antennas, dtype=np.float64)
    coordinates = (indices - (number_of_antennas - 1) / 2.0) * spacing_m
    return coordinates[:, None] * axis_unit[None, :]


def centered_ura_offsets(
    rows: int,
    columns: int,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    horizontal_axis: FloatArray,
    vertical_axis: FloatArray,
) -> FloatArray:
    """Return row-major URA offsets with rows along the vertical axis."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if horizontal_spacing_m <= 0.0 or vertical_spacing_m <= 0.0:
        raise ValueError("URA spacings must be positive")

    horizontal = unit_vector(horizontal_axis, "horizontal_axis")
    vertical = unit_vector(vertical_axis, "vertical_axis")
    if not math.isclose(
        float(np.dot(horizontal, vertical)),
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("horizontal_axis and vertical_axis must be orthogonal")

    column_values = (
        np.arange(columns, dtype=np.float64) - (columns - 1) / 2.0
    ) * horizontal_spacing_m
    row_values = (
        np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0
    ) * vertical_spacing_m

    offsets = np.empty((rows * columns, 3), dtype=np.float64)
    index = 0
    for row_value in row_values:
        for column_value in column_values:
            offsets[index] = (
                column_value * horizontal + row_value * vertical
            )
            index += 1
    return offsets


def verify_surface_orientation(
    horizontal_axis: FloatArray,
    vertical_axis: FloatArray,
    normal: FloatArray,
) -> None:
    horizontal = unit_vector(horizontal_axis, "horizontal_axis")
    vertical = unit_vector(vertical_axis, "vertical_axis")
    normal_unit = unit_vector(normal, "normal")
    matrix = np.stack([horizontal, vertical, normal_unit], axis=0)
    gram = matrix @ matrix.T
    if not np.allclose(gram, np.eye(3), atol=1.0e-12, rtol=0.0):
        raise ValueError("surface axes and normal must be mutually orthogonal")


def unit_norm_plane_wave_response(
    element_offsets_m: FloatArray,
    propagation_direction: FloatArray,
    wavelength_m: float,
) -> ComplexArray:
    """Use exp(-j k u^T r) with offsets measured from the array centre."""
    offsets = np.asarray(element_offsets_m, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("element_offsets_m must have shape (elements, 3)")
    if offsets.shape[0] <= 0 or not np.all(np.isfinite(offsets)):
        raise ValueError("element_offsets_m must be finite and non-empty")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")

    direction = unit_vector(propagation_direction, "propagation_direction")
    wave_number = 2.0 * np.pi / wavelength_m
    phases = -wave_number * (offsets @ direction)
    response = np.exp(1j * phases) / np.sqrt(offsets.shape[0])
    return response.astype(np.complex64)


def bs_to_surface_los_matrix(
    bs_offsets_m: FloatArray,
    surface_offsets_m: FloatArray,
    bs_to_surface_direction: FloatArray,
    centre_distance_m: float,
    wavelength_m: float,
    centre_path_phase_mode: str,
) -> ComplexArray:
    if centre_distance_m <= 0.0:
        raise ValueError("centre_distance_m must be positive")
    direction = unit_vector(
        bs_to_surface_direction,
        "bs_to_surface_direction",
    )
    bs_response = unit_norm_plane_wave_response(
        bs_offsets_m,
        direction,
        wavelength_m,
    )
    surface_response = unit_norm_plane_wave_response(
        surface_offsets_m,
        direction,
        wavelength_m,
    )
    phase = _centre_phase(
        centre_path_phase_mode,
        centre_distance_m,
        wavelength_m,
        channel_vector_hermitian=False,
    )
    scale = np.sqrt(bs_response.size * surface_response.size)
    matrix = (
        phase
        * scale
        * surface_response[:, None]
        * np.conj(bs_response[None, :])
    )
    return matrix.astype(np.complex64)


def surface_to_user_los_vector(
    surface_offsets_m: FloatArray,
    surface_to_user_direction: FloatArray,
    centre_distance_m: float,
    wavelength_m: float,
    centre_path_phase_mode: str,
) -> ComplexArray:
    """Return h such that h^H is the physical surface-to-user channel."""
    if centre_distance_m <= 0.0:
        raise ValueError("centre_distance_m must be positive")
    response = unit_norm_plane_wave_response(
        surface_offsets_m,
        surface_to_user_direction,
        wavelength_m,
    )
    phase = _centre_phase(
        centre_path_phase_mode,
        centre_distance_m,
        wavelength_m,
        channel_vector_hermitian=True,
    )
    vector = phase * np.sqrt(response.size) * response
    return vector.astype(np.complex64)


def _centre_phase(
    mode: str,
    centre_distance_m: float,
    wavelength_m: float,
    channel_vector_hermitian: bool,
) -> complex:
    if mode == "omit":
        return 1.0 + 0.0j
    if mode == "physical_center_distance":
        sign = 1.0 if channel_vector_hermitian else -1.0
        return complex(
            np.exp(
                1j
                * sign
                * 2.0
                * np.pi
                * centre_distance_m
                / wavelength_m
            )
        )
    raise ValueError(
        "centre_path_phase_mode must be 'omit' or "
        "'physical_center_distance'"
    )

