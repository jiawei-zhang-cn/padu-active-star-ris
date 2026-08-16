"""Strict loading and validation for experiment configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arrays import verify_surface_orientation
from .mobility import Rectangle
from .units import integer_channel_uses, wavelength_m


def _exact_keys(
    mapping: dict[str, Any],
    required: set[str],
    location: str,
) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise ValueError(
            f"{location} keys are invalid; missing={missing}, unknown={unknown}"
        )


def _required(value: Any, location: str) -> Any:
    if value is None:
        raise ValueError(f"{location} must be set")
    return value


def _positive_float(value: Any, location: str) -> float:
    number = float(_required(value, location))
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{location} must be finite and positive")
    return number


def _nonnegative_float(value: Any, location: str) -> float:
    number = float(_required(value, location))
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{location} must be finite and non-negative")
    return number


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a positive integer")
    number = int(_required(value, location))
    if number <= 0 or number != value:
        raise ValueError(f"{location} must be a positive integer")
    return number


def _vector3(value: Any, location: str) -> np.ndarray:
    vector = np.asarray(_required(value, location), dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{location} must contain three finite values")
    return vector


def _rectangle(value: Any, location: str) -> Rectangle:
    mapping = _required(value, location)
    if not isinstance(mapping, dict):
        raise ValueError(f"{location} must be an object")
    _exact_keys(mapping, {"x_min", "x_max", "y_min", "y_max"}, location)
    return Rectangle(
        x_min=float(mapping["x_min"]),
        x_max=float(mapping["x_max"]),
        y_min=float(mapping["y_min"]),
        y_max=float(mapping["y_max"]),
    )


@dataclass(frozen=True)
class ArrayConfig:
    bs_antennas: int
    star_rows: int
    star_columns: int
    bs_spacing_wavelengths: float
    star_horizontal_spacing_wavelengths: float
    star_vertical_spacing_wavelengths: float
    bs_ula_axis: np.ndarray
    star_horizontal_axis: np.ndarray
    star_vertical_axis: np.ndarray
    star_normal: np.ndarray

    @property
    def star_elements(self) -> int:
        return self.star_rows * self.star_columns


@dataclass(frozen=True)
class GeometryConfig:
    bs_position_m: np.ndarray
    star_center_position_m: np.ndarray
    user_height_m: float
    reflection_region: Rectangle
    transmission_region: Rectangle


@dataclass(frozen=True)
class ChannelConfig:
    carrier_frequency_ghz: float
    time_slot_s: float
    bs_star_rician_factor_db: float
    user_rician_factor_db: tuple[float, ...]
    center_path_phase_mode: str
    path_loss_model: str


@dataclass(frozen=True)
class MobilityConfig:
    user_sides: tuple[str, ...]
    user_speeds_m_per_s: tuple[float, ...]
    arrival_tolerance_m: float


@dataclass(frozen=True)
class NoiseConfig:
    noise_psd_dbm_per_hz: float
    bandwidth_hz: float
    receiver_noise_figure_db: float
    star_noise_figure_db: float


@dataclass(frozen=True)
class PowerConfig:
    bs_max_output_dbm: float
    bs_per_antenna_max_output_dbm: float | None
    star_total_max_output_dbm: float
    star_per_element_max_output_dbm: float
    star_max_power_gain_db: float


@dataclass(frozen=True)
class FiniteBlocklengthConfig:
    packet_duration_s: float
    decoding_error_probability: float
    blocklength: int
    minimum_packet_payload_bits_per_user: tuple[float, ...]
    optimization_design_payload_bits_per_user: tuple[float, ...]

    @property
    def minimum_spectral_efficiency_per_user(self) -> tuple[float, ...]:
        return tuple(
            payload_bits / self.blocklength
            for payload_bits in self.minimum_packet_payload_bits_per_user
        )

    @property
    def optimization_design_spectral_efficiency_per_user(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            payload_bits / self.blocklength
            for payload_bits in self.optimization_design_payload_bits_per_user
        )


@dataclass(frozen=True)
class LearningConfig:
    history_length: int


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    array: ArrayConfig
    geometry: GeometryConfig
    channel: ChannelConfig
    mobility: MobilityConfig
    noise: NoiseConfig
    power: PowerConfig
    finite_blocklength: FiniteBlocklengthConfig
    learning: LearningConfig

    @property
    def number_of_users(self) -> int:
        return len(self.mobility.user_sides)


ROOT_KEYS = {
    "schema_version",
    "array",
    "geometry",
    "channel",
    "mobility",
    "noise",
    "power",
    "finite_blocklength",
    "learning",
}

SECTION_KEYS = {
    "array": {
        "bs_antennas",
        "star_rows",
        "star_columns",
        "bs_spacing_wavelengths",
        "star_horizontal_spacing_wavelengths",
        "star_vertical_spacing_wavelengths",
        "bs_ula_axis",
        "star_horizontal_axis",
        "star_vertical_axis",
        "star_normal",
    },
    "geometry": {
        "bs_position_m",
        "star_center_position_m",
        "user_height_m",
        "reflection_region_xy_m",
        "transmission_region_xy_m",
    },
    "channel": {
        "carrier_frequency_ghz",
        "time_slot_s",
        "bs_star_rician_factor_db",
        "user_rician_factor_db",
        "center_path_phase_mode",
        "path_loss_model",
    },
    "mobility": {
        "user_sides",
        "user_speeds_m_per_s",
        "arrival_tolerance_m",
    },
    "noise": {
        "noise_psd_dbm_per_hz",
        "bandwidth_hz",
        "receiver_noise_figure_db",
        "star_noise_figure_db",
    },
    "power": {
        "bs_max_output_dbm",
        "bs_per_antenna_max_output_dbm",
        "star_total_max_output_dbm",
        "star_per_element_max_output_dbm",
        "star_max_power_gain_db",
    },
    "finite_blocklength": {
        "packet_duration_s",
        "decoding_error_probability",
        "minimum_packet_payload_bits_per_user",
        "optimization_design_payload_bits_per_user",
    },
    "learning": {
        "history_length",
    },
}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    _exact_keys(raw, ROOT_KEYS, "root")
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must equal 1")

    sections: dict[str, dict[str, Any]] = {}
    for section_name, keys in SECTION_KEYS.items():
        section = raw[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be an object")
        _exact_keys(section, keys, section_name)
        sections[section_name] = section

    array_raw = sections["array"]
    array = ArrayConfig(
        bs_antennas=_positive_int(
            array_raw["bs_antennas"],
            "array.bs_antennas",
        ),
        star_rows=_positive_int(array_raw["star_rows"], "array.star_rows"),
        star_columns=_positive_int(
            array_raw["star_columns"],
            "array.star_columns",
        ),
        bs_spacing_wavelengths=_positive_float(
            array_raw["bs_spacing_wavelengths"],
            "array.bs_spacing_wavelengths",
        ),
        star_horizontal_spacing_wavelengths=_positive_float(
            array_raw["star_horizontal_spacing_wavelengths"],
            "array.star_horizontal_spacing_wavelengths",
        ),
        star_vertical_spacing_wavelengths=_positive_float(
            array_raw["star_vertical_spacing_wavelengths"],
            "array.star_vertical_spacing_wavelengths",
        ),
        bs_ula_axis=_vector3(array_raw["bs_ula_axis"], "array.bs_ula_axis"),
        star_horizontal_axis=_vector3(
            array_raw["star_horizontal_axis"],
            "array.star_horizontal_axis",
        ),
        star_vertical_axis=_vector3(
            array_raw["star_vertical_axis"],
            "array.star_vertical_axis",
        ),
        star_normal=_vector3(
            array_raw["star_normal"],
            "array.star_normal",
        ),
    )
    verify_surface_orientation(
        array.star_horizontal_axis,
        array.star_vertical_axis,
        array.star_normal,
    )

    geometry_raw = sections["geometry"]
    geometry = GeometryConfig(
        bs_position_m=_vector3(
            geometry_raw["bs_position_m"],
            "geometry.bs_position_m",
        ),
        star_center_position_m=_vector3(
            geometry_raw["star_center_position_m"],
            "geometry.star_center_position_m",
        ),
        user_height_m=_positive_float(
            geometry_raw["user_height_m"],
            "geometry.user_height_m",
        ),
        reflection_region=_rectangle(
            geometry_raw["reflection_region_xy_m"],
            "geometry.reflection_region_xy_m",
        ),
        transmission_region=_rectangle(
            geometry_raw["transmission_region_xy_m"],
            "geometry.transmission_region_xy_m",
        ),
    )

    channel_raw = sections["channel"]
    user_rician = tuple(
        float(value)
        for value in _required(
            channel_raw["user_rician_factor_db"],
            "channel.user_rician_factor_db",
        )
    )
    if not user_rician or not np.all(np.isfinite(user_rician)):
        raise ValueError(
            "channel.user_rician_factor_db must contain finite values"
        )
    center_phase_mode = str(
        _required(
            channel_raw["center_path_phase_mode"],
            "channel.center_path_phase_mode",
        )
    )
    if center_phase_mode not in {"omit", "physical_center_distance"}:
        raise ValueError(
            "channel.center_path_phase_mode must be 'omit' or "
            "'physical_center_distance'"
        )
    if channel_raw["path_loss_model"] != "3gpp_umi_street_canyon_los":
        raise ValueError(
            "channel.path_loss_model must be "
            "'3gpp_umi_street_canyon_los'"
        )
    channel = ChannelConfig(
        carrier_frequency_ghz=_positive_float(
            channel_raw["carrier_frequency_ghz"],
            "channel.carrier_frequency_ghz",
        ),
        time_slot_s=_positive_float(
            channel_raw["time_slot_s"],
            "channel.time_slot_s",
        ),
        bs_star_rician_factor_db=float(
            _required(
                channel_raw["bs_star_rician_factor_db"],
                "channel.bs_star_rician_factor_db",
            )
        ),
        user_rician_factor_db=user_rician,
        center_path_phase_mode=center_phase_mode,
        path_loss_model=channel_raw["path_loss_model"],
    )

    mobility_raw = sections["mobility"]
    user_sides = tuple(
        str(value)
        for value in _required(
            mobility_raw["user_sides"],
            "mobility.user_sides",
        )
    )
    if not user_sides or any(side not in {"R", "T"} for side in user_sides):
        raise ValueError("mobility.user_sides entries must be 'R' or 'T'")
    if set(user_sides) != {"R", "T"}:
        raise ValueError(
            "mobility.user_sides must include at least one reflection-side "
            "and one transmission-side user"
        )
    user_speeds = tuple(
        float(value)
        for value in _required(
            mobility_raw["user_speeds_m_per_s"],
            "mobility.user_speeds_m_per_s",
        )
    )
    if len(user_speeds) != len(user_sides):
        raise ValueError(
            "mobility.user_speeds_m_per_s and user_sides must have equal length"
        )
    if any(not np.isfinite(speed) or speed < 0.0 for speed in user_speeds):
        raise ValueError("all user speeds must be finite and non-negative")
    if len(user_rician) != len(user_sides):
        raise ValueError(
            "channel.user_rician_factor_db must contain one value per user"
        )
    mobility = MobilityConfig(
        user_sides=user_sides,
        user_speeds_m_per_s=user_speeds,
        arrival_tolerance_m=_positive_float(
            mobility_raw["arrival_tolerance_m"],
            "mobility.arrival_tolerance_m",
        ),
    )

    noise_raw = sections["noise"]
    noise = NoiseConfig(
        noise_psd_dbm_per_hz=float(
            _required(
                noise_raw["noise_psd_dbm_per_hz"],
                "noise.noise_psd_dbm_per_hz",
            )
        ),
        bandwidth_hz=_positive_float(
            noise_raw["bandwidth_hz"],
            "noise.bandwidth_hz",
        ),
        receiver_noise_figure_db=_nonnegative_float(
            noise_raw["receiver_noise_figure_db"],
            "noise.receiver_noise_figure_db",
        ),
        star_noise_figure_db=_nonnegative_float(
            noise_raw["star_noise_figure_db"],
            "noise.star_noise_figure_db",
        ),
    )

    power_raw = sections["power"]
    per_antenna_power_raw = power_raw["bs_per_antenna_max_output_dbm"]
    if per_antenna_power_raw is not None:
        per_antenna_power_raw = float(per_antenna_power_raw)
        if not np.isfinite(per_antenna_power_raw):
            raise ValueError(
                "power.bs_per_antenna_max_output_dbm must be finite or null"
            )
    power = PowerConfig(
        bs_max_output_dbm=float(
            _required(
                power_raw["bs_max_output_dbm"],
                "power.bs_max_output_dbm",
            )
        ),
        bs_per_antenna_max_output_dbm=per_antenna_power_raw,
        star_total_max_output_dbm=float(
            _required(
                power_raw["star_total_max_output_dbm"],
                "power.star_total_max_output_dbm",
            )
        ),
        star_per_element_max_output_dbm=float(
            _required(
                power_raw["star_per_element_max_output_dbm"],
                "power.star_per_element_max_output_dbm",
            )
        ),
        star_max_power_gain_db=_positive_float(
            power_raw["star_max_power_gain_db"],
            "power.star_max_power_gain_db",
        ),
    )

    fbl_raw = sections["finite_blocklength"]
    packet_duration = _positive_float(
        fbl_raw["packet_duration_s"],
        "finite_blocklength.packet_duration_s",
    )
    if packet_duration > channel.time_slot_s:
        raise ValueError(
            "finite_blocklength.packet_duration_s must not exceed "
            "channel.time_slot_s"
        )
    error_probability = float(
        _required(
            fbl_raw["decoding_error_probability"],
            "finite_blocklength.decoding_error_probability",
        )
    )
    if not 0.0 < error_probability < 0.5:
        raise ValueError(
            "finite_blocklength.decoding_error_probability must lie in (0, 0.5)"
        )
    finite_blocklength = FiniteBlocklengthConfig(
        packet_duration_s=packet_duration,
        decoding_error_probability=error_probability,
        blocklength=integer_channel_uses(
            noise.bandwidth_hz,
            packet_duration,
        ),
        minimum_packet_payload_bits_per_user=tuple(
            _nonnegative_float(
                value,
                "finite_blocklength.minimum_packet_payload_bits_per_user",
            )
            for value in _required(
                fbl_raw["minimum_packet_payload_bits_per_user"],
                "finite_blocklength.minimum_packet_payload_bits_per_user",
            )
        ),
        optimization_design_payload_bits_per_user=tuple(
            _nonnegative_float(
                value,
                "finite_blocklength.optimization_design_payload_bits_per_user",
            )
            for value in _required(
                fbl_raw["optimization_design_payload_bits_per_user"],
                "finite_blocklength.optimization_design_payload_bits_per_user",
            )
        ),
    )
    if (
        len(finite_blocklength.minimum_packet_payload_bits_per_user)
        != len(mobility.user_sides)
    ):
        raise ValueError(
            "finite_blocklength.minimum_packet_payload_bits_per_user "
            "must have one value per user"
        )
    if (
        len(finite_blocklength.optimization_design_payload_bits_per_user)
        != len(mobility.user_sides)
    ):
        raise ValueError(
            "finite_blocklength.optimization_design_payload_bits_per_user "
            "must have one value per user"
        )
    if any(
        design_payload < business_payload
        for business_payload, design_payload in zip(
            finite_blocklength.minimum_packet_payload_bits_per_user,
            finite_blocklength.optimization_design_payload_bits_per_user,
            strict=True,
        )
    ):
        raise ValueError(
            "finite_blocklength.optimization_design_payload_bits_per_user "
            "must be no smaller than the corresponding business minimum"
        )

    learning_raw = sections["learning"]
    learning = LearningConfig(
        history_length=_positive_int(
            learning_raw["history_length"],
            "learning.history_length",
        ),
    )

    wavelength = wavelength_m(channel.carrier_frequency_ghz)
    minimum_surface_height = (
        geometry.star_center_position_m[2]
        - (array.star_rows - 1)
        * array.star_vertical_spacing_wavelengths
        * wavelength
        / 2.0
    )
    if minimum_surface_height <= 0.0:
        raise ValueError("the lowest active STAR-RIS element is not above ground")

    signed_bs_side = float(
        np.dot(
            geometry.bs_position_m - geometry.star_center_position_m,
            array.star_normal,
        )
    )
    if signed_bs_side >= 0.0:
        raise ValueError(
            "the BS must lie on the negative side of star_normal; "
            "that side is the reflection side"
        )
    _validate_region_side(
        geometry.reflection_region,
        geometry.user_height_m,
        geometry.star_center_position_m,
        array.star_normal,
        expected_sign=-1,
        name="reflection_region_xy_m",
    )
    _validate_region_side(
        geometry.transmission_region,
        geometry.user_height_m,
        geometry.star_center_position_m,
        array.star_normal,
        expected_sign=1,
        name="transmission_region_xy_m",
    )
    _validate_umi_los_applicability(
        geometry=geometry,
        carrier_frequency_ghz=channel.carrier_frequency_ghz,
    )

    return ExperimentConfig(
        schema_version=1,
        array=array,
        geometry=geometry,
        channel=channel,
        mobility=mobility,
        noise=noise,
        power=power,
        finite_blocklength=finite_blocklength,
        learning=learning,
    )


def _validate_region_side(
    region: Rectangle,
    user_height_m: float,
    surface_center: np.ndarray,
    surface_normal: np.ndarray,
    expected_sign: int,
    name: str,
) -> None:
    corners = (
        (region.x_min, region.y_min),
        (region.x_min, region.y_max),
        (region.x_max, region.y_min),
        (region.x_max, region.y_max),
    )
    signed_values = [
        float(
            np.dot(
                np.array([x, y, user_height_m]) - surface_center,
                surface_normal,
            )
        )
        for x, y in corners
    ]
    if expected_sign < 0 and not all(value < 0.0 for value in signed_values):
        raise ValueError(f"geometry.{name} must lie strictly on the reflection side")
    if expected_sign > 0 and not all(value > 0.0 for value in signed_values):
        raise ValueError(
            f"geometry.{name} must lie strictly on the transmission side"
        )


def _horizontal_distance_to_rectangle(
    point_xy: np.ndarray,
    rectangle: Rectangle,
) -> tuple[float, float]:
    nearest_x = float(
        np.clip(point_xy[0], rectangle.x_min, rectangle.x_max)
    )
    nearest_y = float(
        np.clip(point_xy[1], rectangle.y_min, rectangle.y_max)
    )
    minimum = float(
        np.linalg.norm(
            point_xy - np.array([nearest_x, nearest_y])
        )
    )
    corners = np.array(
        [
            [rectangle.x_min, rectangle.y_min],
            [rectangle.x_min, rectangle.y_max],
            [rectangle.x_max, rectangle.y_min],
            [rectangle.x_max, rectangle.y_max],
        ],
        dtype=np.float64,
    )
    maximum = float(
        np.max(np.linalg.norm(corners - point_xy, axis=1))
    )
    return minimum, maximum


def _validate_umi_los_applicability(
    geometry: GeometryConfig,
    carrier_frequency_ghz: float,
) -> None:
    if not 0.5 < carrier_frequency_ghz < 100.0:
        raise ValueError(
            "3GPP UMi Street Canyon LoS requires carrier frequency "
            "strictly between 0.5 GHz and 100 GHz"
        )

    bs_star_distance_2d = float(
        np.linalg.norm(
            geometry.bs_position_m[:2]
            - geometry.star_center_position_m[:2]
        )
    )
    distance_ranges = {
        "BS-to-STAR-RIS": (
            bs_star_distance_2d,
            bs_star_distance_2d,
        ),
        "STAR-RIS-to-reflection-region": (
            _horizontal_distance_to_rectangle(
                geometry.star_center_position_m[:2],
                geometry.reflection_region,
            )
        ),
        "STAR-RIS-to-transmission-region": (
            _horizontal_distance_to_rectangle(
                geometry.star_center_position_m[:2],
                geometry.transmission_region,
            )
        ),
    }
    for link_name, (minimum, maximum) in distance_ranges.items():
        if minimum < 10.0 or maximum > 5_000.0:
            raise ValueError(
                f"{link_name} horizontal distance range "
                f"[{minimum}, {maximum}] m violates the 3GPP UMi "
                "Street Canyon LoS range [10, 5000] m"
            )
