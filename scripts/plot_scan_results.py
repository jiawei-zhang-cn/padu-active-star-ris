"""Plot five-method PADU scan results as IEEE-style PDFs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except (ImportError, OSError, ValueError) as error:
    raise SystemExit(
        "Plotting dependencies could not be loaded in the current Python "
        "environment. Run this script in the tested drl environment, "
        "for example: conda run --no-capture-output -n drl python "
        "scripts/plot_scan_results.py ..."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "results" / "figures_10seed"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    key: str
    csv_name: str
    color: str
    linestyle: str
    marker: str


SPEED_METHODS = (
    MethodSpec(
        label="PADU",
        key="padu",
        csv_name="padu.csv",
        color="#0072B2",
        linestyle="-",
        marker="o",
    ),
    MethodSpec(
        label="Benchmark 1",
        key="benchmark1_without_csi_prediction",
        csv_name="benchmark1_without_csi_prediction.csv",
        color="#D55E00",
        linestyle="--",
        marker="s",
    ),
    MethodSpec(
        label="Benchmark 2",
        key="benchmark2_without_unfolding_refinement",
        csv_name="benchmark2_without_unfolding_refinement.csv",
        color="#009E73",
        linestyle="-.",
        marker="^",
    ),
    MethodSpec(
        label="Benchmark 3",
        key="benchmark3_perfect_next_slot_csi",
        csv_name="benchmark3_perfect_next_slot_csi.csv",
        color="#CC79A7",
        linestyle=":",
        marker="D",
    ),
    MethodSpec(
        label="Benchmark 4",
        key="benchmark4_nmse_trained_point_gru",
        csv_name="benchmark4_nmse_trained_point_gru.csv",
        color="#666666",
        linestyle="-",
        marker="v",
    ),
)

RICIAN_METHODS = (
    MethodSpec(
        label="PADU",
        key="padu",
        csv_name="padu.csv",
        color="#0072B2",
        linestyle="-",
        marker="o",
    ),
    MethodSpec(
        label="Benchmark 1",
        key="benchmark1_without_csi_prediction",
        csv_name="benchmark1_without_csi_prediction.csv",
        color="#D55E00",
        linestyle="--",
        marker="s",
    ),
    MethodSpec(
        label="Benchmark 2",
        key="benchmark2_without_unfolding_refinement",
        csv_name="benchmark2_without_unfolding_refinement.csv",
        color="#009E73",
        linestyle="-.",
        marker="^",
    ),
    MethodSpec(
        label="Benchmark 3",
        key="benchmark3_perfect_next_slot_csi",
        csv_name="benchmark3_perfect_next_slot_csi.csv",
        color="#CC79A7",
        linestyle=":",
        marker="D",
    ),
    MethodSpec(
        label="Benchmark 4",
        key="benchmark4_nmse_trained_point_gru",
        csv_name="benchmark4_nmse_trained_point_gru.csv",
        color="#666666",
        linestyle="-",
        marker="v",
    ),
)

AXIS_CONFIG = {
    "speed": {
        "methods": SPEED_METHODS,
        "value_column": "speed_m_per_s",
        "expected_values": (0.3, 0.6, 0.9, 1.2, 1.5, 1.8),
        "xlabel": r"User Speed $v$ (m/s)",
        "output_stem": "speed",
        "outage_legend_loc": "upper left",
        "throughput_legend_loc": "lower left",
    },
    "rician": {
        "methods": RICIAN_METHODS,
        "value_column": "rician_factor_db",
        "expected_values": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        "xlabel": r"Rician Factor $\kappa$ (dB)",
        "output_stem": "rician",
        "outage_legend_loc": "upper right",
        "throughput_legend_loc": "upper left",
    },
}

COMPLETED_SLOTS_PER_SEED = 192

PDF_METADATA = {
    "Title": None,
    "Author": None,
    "Subject": None,
    "Keywords": None,
    "Creator": None,
    "Producer": None,
    "CreationDate": None,
    "ModDate": None,
    "Trapped": None,
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PADU scan results."
    )
    parser.add_argument(
        "--scan-axis",
        choices=tuple(AXIS_CONFIG),
        default="speed",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        help=(
            "Directory containing method CSV files or method scan output "
            "directories. Defaults to results/scans_10seed/<scan-axis>."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--expected-seed-count",
        type=int,
        help="Require every plotted CSV row to use this many evaluation seeds.",
    )
    return parser.parse_args()


def _load_method(
    spec: MethodSpec,
    *,
    csv_path: Path,
    value_column: str,
    expected_values: tuple[float, ...],
    expected_seed_count: int | None,
) -> tuple[list[dict[str, float]], int]:
    required_columns = {
        "domain",
        value_column,
        "evaluation_seed_count",
        "completed_slots",
        "joint_qos_outage_count",
        "joint_qos_outage_rate",
        "mean_seed_joint_qos_effective_throughput_bps",
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV file: {csv_path}")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {sorted(missing)}"
            )
        rows = list(reader)

    if len(rows) != len(expected_values):
        raise ValueError(
            f"{csv_path} has {len(rows)} rows; expected {len(expected_values)}"
        )

    parsed: list[dict[str, float]] = []
    observed_seed_count: int | None = None
    for row in rows:
        if row["domain"] != "in_domain":
            raise ValueError(f"unexpected domain in {csv_path}: {row['domain']}")
        scan_value = float(row[value_column])
        seed_count = int(row["evaluation_seed_count"])
        completed_slots = int(row["completed_slots"])
        outage_count = int(row["joint_qos_outage_count"])
        outage_rate = float(row["joint_qos_outage_rate"])
        effective_throughput_bps = float(
            row["mean_seed_joint_qos_effective_throughput_bps"]
        )
        if observed_seed_count is None:
            observed_seed_count = seed_count
        elif seed_count != observed_seed_count:
            raise ValueError(
                f"inconsistent evaluation_seed_count in {csv_path}: {seed_count}"
            )
        if expected_seed_count is not None and seed_count != expected_seed_count:
            raise ValueError(
                f"unexpected evaluation_seed_count in {csv_path}: {seed_count}"
            )
        expected_completed_slots = seed_count * COMPLETED_SLOTS_PER_SEED
        if completed_slots != expected_completed_slots:
            raise ValueError(
                f"unexpected completed_slots in {csv_path}: {completed_slots}"
            )
        exact_rate = outage_count / completed_slots
        if not math.isclose(outage_rate, exact_rate, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"outage count/rate mismatch in {csv_path} at {scan_value}"
            )
        if not math.isfinite(effective_throughput_bps):
            raise ValueError(
                f"non-finite effective throughput in {csv_path} at {scan_value}"
            )
        parsed.append(
            {
                "scan_value": scan_value,
                "outage_percent": 100.0 * exact_rate,
                "effective_throughput_mbps": effective_throughput_bps / 1e6,
            }
        )

    parsed.sort(key=lambda item: item["scan_value"])
    values = tuple(item["scan_value"] for item in parsed)
    if values != expected_values:
        raise ValueError(
            f"unexpected scan grid in {csv_path}: {values}"
        )
    if observed_seed_count is None:
        raise ValueError(f"CSV file has no data rows: {csv_path}")
    return parsed, observed_seed_count


def _resolve_method_csv(
    spec: MethodSpec,
    *,
    scan_root: Path,
    scan_axis: str,
) -> Path:
    if not scan_root.exists():
        raise FileNotFoundError(
            f"scan root does not exist: {scan_root}. "
            "Run scripts/run_ten_seed_scans.ps1 before plotting 10-seed scans."
        )
    direct_path = scan_root / spec.csv_name
    if direct_path.is_file():
        return direct_path
    nested_path = scan_root / spec.key / f"{scan_axis}_scan_summary.csv"
    if nested_path.is_file():
        return nested_path
    raise FileNotFoundError(
        "missing scan CSV for "
        f"{spec.label}; checked {direct_path} and {nested_path}"
    )


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0.0 or not math.isfinite(raw_step):
        raise ValueError(f"invalid raw step: {raw_step}")
    exponent = math.floor(math.log10(raw_step))
    base = raw_step / (10.0**exponent)
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if base <= multiplier:
            return multiplier * (10.0**exponent)
    raise AssertionError("unreachable")


def _auto_axis_limits(
    data: dict[str, list[dict[str, float]]],
    metric_key: str,
) -> tuple[tuple[float, float], tuple[float, ...]]:
    values = [
        row[metric_key]
        for rows in data.values()
        for row in rows
    ]
    if not values:
        raise ValueError("cannot set axis limits without data")
    minimum = min(values)
    maximum = max(values)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("cannot set axis limits from non-finite data")
    if math.isclose(minimum, maximum):
        padding = max(abs(maximum) * 0.05, 1.0)
    else:
        padding = 0.06 * (maximum - minimum)
    padded_minimum = minimum - padding
    padded_maximum = maximum + padding
    step = _nice_step((padded_maximum - padded_minimum) / 5.0)
    if metric_key == "outage_percent":
        lower = 0.0
    elif metric_key == "effective_throughput_mbps":
        lower = math.floor(padded_minimum / step) * step
    else:
        raise ValueError(f"unsupported metric: {metric_key}")
    upper = math.ceil(padded_maximum / step) * step
    ticks = []
    value = lower
    for _ in range(32):
        ticks.append(round(value, 10))
        if value >= upper:
            break
        value += step
    if ticks[-1] < upper:
        ticks.append(round(upper, 10))
    return (lower, upper), tuple(ticks)


def _metric_values_in_range(
    data: dict[str, list[dict[str, float]]],
    *,
    metric_key: str,
    xlim: tuple[float, float],
) -> list[float]:
    lower, upper = xlim
    return [
        row[metric_key]
        for rows in data.values()
        for row in rows
        if lower <= row["scan_value"] <= upper
    ]


def _add_rician_outage_inset(
    axis,
    data: dict[str, list[dict[str, float]]],
    *,
    methods: tuple[MethodSpec, ...],
) -> None:
    xlim = (4.0, 7.0)
    values = _metric_values_in_range(
        data,
        metric_key="outage_percent",
        xlim=xlim,
    )
    if not values:
        raise ValueError("Rician outage inset has no values in 4--7 dB range")
    maximum = max(values)
    if maximum <= 0.0:
        upper = 0.5
        yticks = (0.0, 0.25, 0.5)
    else:
        upper = math.ceil((maximum * 1.20) / 0.25) * 0.25
        upper = max(upper, 0.5)
        yticks = tuple(
            value
            for value in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
            if value <= upper
        )
        if yticks[-1] < upper:
            yticks = (*yticks, upper)
    inset = axis.inset_axes([0.52, 0.16, 0.43, 0.38])
    inset.set_facecolor("white")
    for spec in methods:
        rows = [
            row for row in data[spec.label]
            if xlim[0] <= row["scan_value"] <= xlim[1]
        ]
        inset.plot(
            [row["scan_value"] for row in rows],
            [row["outage_percent"] for row in rows],
            color=spec.color,
            linestyle=spec.linestyle,
            marker=spec.marker,
            markerfacecolor="white",
            markeredgecolor=spec.color,
            markeredgewidth=0.75,
            linewidth=0.95,
            markersize=3.1,
            zorder=3,
        )
    inset.set_xlim(*xlim)
    inset.set_ylim(0.0, upper)
    inset.set_xticks((4.0, 5.0, 6.0, 7.0))
    inset.set_yticks(yticks)
    inset.tick_params(direction="out", labelsize=7.8, pad=1.0)
    inset.grid(axis="y", color="#E0E0E0", linewidth=0.45, zorder=0)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
    inset.set_title("4-7 dB", fontsize=8.6, pad=1.5)


def _plot_metric(
    data: dict[str, list[dict[str, float]]],
    *,
    scan_axis: str,
    methods: tuple[MethodSpec, ...],
    expected_values: tuple[float, ...],
    xlabel: str,
    legend_loc: str,
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(3.5, 2.65))
    for spec in methods:
        rows = data[spec.label]
        axis.plot(
            [row["scan_value"] for row in rows],
            [row[metric_key] for row in rows],
            label=spec.label,
            color=spec.color,
            linestyle=spec.linestyle,
            marker=spec.marker,
            markerfacecolor="white",
            markeredgecolor=spec.color,
            markeredgewidth=0.9,
            zorder=3,
        )

    axis.set_xlabel(xlabel, fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.set_xticks(expected_values)
    axis.tick_params(direction="out", labelsize=10)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    legend = axis.legend(
        loc=legend_loc,
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=7.8,
        handlelength=2.3,
        handletextpad=0.45,
        columnspacing=0.9,
        borderpad=0.35,
        borderaxespad=0.35,
    )
    legend.get_frame().set_linewidth(0.6)

    if metric_key not in {"outage_percent", "effective_throughput_mbps"}:
        raise ValueError(f"unsupported metric: {metric_key}")
    ylim, yticks = _auto_axis_limits(data, metric_key)
    axis.set_ylim(*ylim)
    axis.set_yticks(yticks)
    if scan_axis == "rician" and metric_key == "outage_percent":
        _add_rician_outage_inset(axis, data, methods=methods)

    figure.subplots_adjust(left=0.235, right=0.985, bottom=0.235, top=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        metadata=PDF_METADATA,
    )
    plt.close(figure)


def main() -> None:
    arguments = _parse_arguments()
    _configure_matplotlib()
    config = AXIS_CONFIG[arguments.scan_axis]
    methods = config["methods"]
    expected_values = config["expected_values"]
    scan_root = (
        arguments.scan_root
        if arguments.scan_root is not None
        else ROOT / "results" / "scans_10seed" / arguments.scan_axis
    )
    data: dict[str, list[dict[str, float]]] = {}
    seed_counts: dict[str, int] = {}
    for spec in methods:
        csv_path = _resolve_method_csv(
            spec,
            scan_root=scan_root,
            scan_axis=arguments.scan_axis,
        )
        rows, seed_count = _load_method(
            spec,
            csv_path=csv_path,
            value_column=config["value_column"],
            expected_values=expected_values,
            expected_seed_count=arguments.expected_seed_count,
        )
        data[spec.label] = rows
        seed_counts[spec.label] = seed_count
    unique_seed_counts = set(seed_counts.values())
    if len(unique_seed_counts) != 1:
        raise ValueError(f"methods use different seed counts: {seed_counts}")
    output_stem = config["output_stem"]
    _plot_metric(
        data,
        scan_axis=arguments.scan_axis,
        methods=methods,
        expected_values=expected_values,
        xlabel=config["xlabel"],
        legend_loc=config["outage_legend_loc"],
        metric_key="outage_percent",
        ylabel="Joint QoS Outage Probability (%)",
        output_path=(
            arguments.output_directory
            / f"{output_stem}_joint_qos_outage_rate.pdf"
        ),
    )
    _plot_metric(
        data,
        scan_axis=arguments.scan_axis,
        methods=methods,
        expected_values=expected_values,
        xlabel=config["xlabel"],
        legend_loc=config["throughput_legend_loc"],
        metric_key="effective_throughput_mbps",
        ylabel="QoS-Effective Throughput (Mbit/s)",
        output_path=(
            arguments.output_directory
            / f"{output_stem}_joint_qos_effective_throughput.pdf"
        ),
    )
    print(
        (
            arguments.output_directory
            / f"{output_stem}_joint_qos_outage_rate.pdf"
        ).resolve()
    )
    print(
        (
            arguments.output_directory
            / f"{output_stem}_joint_qos_effective_throughput.pdf"
        ).resolve()
    )


if __name__ == "__main__":
    main()
