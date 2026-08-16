"""Paired one-factor scans for PADU evaluation schemes."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

from .unfolding_workflow import evaluate_padu


DEFAULT_SPEED_VALUES = (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)


def build_speed_run_config(
    source: dict[str, Any],
    *,
    root_seed: int,
    speed_m_per_s: float,
) -> dict[str, Any]:
    """Create one paired test config while preserving Rician sampling."""
    if not isinstance(source, dict):
        raise ValueError("run configuration must be an object")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise ValueError("root_seed must be an integer")
    if not math.isfinite(speed_m_per_s) or speed_m_per_s < 0.0:
        raise ValueError("speed_m_per_s must be finite and non-negative")
    target = deepcopy(source)
    target["root_seeds"] = [root_seed]
    try:
        distribution = target["task_distributions"]["in_domain_test"]
        speed_intervals = distribution["user_speed_intervals_m_per_s"]
        rician_intervals = distribution["user_rician_factor_intervals_db"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "run configuration has no valid in_domain_test distribution"
        ) from error
    if (
        not isinstance(speed_intervals, list)
        or not isinstance(rician_intervals, list)
        or not speed_intervals
        or len(speed_intervals) != len(rician_intervals)
    ):
        raise ValueError(
            "in_domain_test speed and Rician intervals must have equal "
            "non-zero length"
        )
    distribution["user_speed_intervals_m_per_s"] = [
        {"minimum": speed_m_per_s, "maximum": speed_m_per_s}
        for _ in speed_intervals
    ]
    return target


def build_rician_run_config(
    source: dict[str, Any],
    *,
    root_seed: int,
    rician_factor_db: float,
) -> dict[str, Any]:
    """Create one paired test config while preserving speed sampling."""
    if not isinstance(source, dict):
        raise ValueError("run configuration must be an object")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise ValueError("root_seed must be an integer")
    if not math.isfinite(rician_factor_db):
        raise ValueError("rician_factor_db must be finite")
    target = deepcopy(source)
    target["root_seeds"] = [root_seed]
    try:
        distribution = target["task_distributions"]["in_domain_test"]
        speed_intervals = distribution["user_speed_intervals_m_per_s"]
        rician_intervals = distribution["user_rician_factor_intervals_db"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "run configuration has no valid in_domain_test distribution"
        ) from error
    if (
        not isinstance(speed_intervals, list)
        or not isinstance(rician_intervals, list)
        or not speed_intervals
        or len(speed_intervals) != len(rician_intervals)
    ):
        raise ValueError(
            "in_domain_test speed and Rician intervals must have equal "
            "non-zero length"
        )
    distribution["user_rician_factor_intervals_db"] = [
        {"minimum": rician_factor_db, "maximum": rician_factor_db}
        for _ in rician_intervals
    ]
    return target


def run_speed_scan(
    *,
    system_config_path: str | Path,
    run_config_path: str | Path,
    unfolding_config_path: str | Path,
    checkpoint_seed_directory: str | Path,
    output_root: str | Path,
    evaluation_seeds: Sequence[int],
    speed_values: Sequence[float] | None = None,
    domains: Sequence[str] = ("in_domain",),
    perfect_next_slot_csi: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run paired PADU speed evaluations and aggregate them."""
    speeds = _validate_speeds(
        DEFAULT_SPEED_VALUES if speed_values is None else speed_values
    )
    return _run_scan(
        scan_axis="speed",
        scan_values=speeds,
        system_config_path=system_config_path,
        run_config_path=run_config_path,
        unfolding_config_path=unfolding_config_path,
        checkpoint_seed_directory=checkpoint_seed_directory,
        output_root=output_root,
        evaluation_seeds=evaluation_seeds,
        domains=domains,
        perfect_next_slot_csi=perfect_next_slot_csi,
        resume=resume,
    )


def run_rician_scan(
    *,
    system_config_path: str | Path,
    run_config_path: str | Path,
    unfolding_config_path: str | Path,
    checkpoint_seed_directory: str | Path,
    output_root: str | Path,
    evaluation_seeds: Sequence[int],
    rician_values: Sequence[float],
    domains: Sequence[str] = ("in_domain",),
    perfect_next_slot_csi: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run paired PADU Rician evaluations and aggregate them."""
    factors = _validate_rician_factors(rician_values)
    return _run_scan(
        scan_axis="rician",
        scan_values=factors,
        system_config_path=system_config_path,
        run_config_path=run_config_path,
        unfolding_config_path=unfolding_config_path,
        checkpoint_seed_directory=checkpoint_seed_directory,
        output_root=output_root,
        evaluation_seeds=evaluation_seeds,
        domains=domains,
        perfect_next_slot_csi=perfect_next_slot_csi,
        resume=resume,
    )


def _run_scan(
    *,
    scan_axis: str,
    scan_values: Sequence[float],
    system_config_path: str | Path,
    run_config_path: str | Path,
    unfolding_config_path: str | Path,
    checkpoint_seed_directory: str | Path,
    output_root: str | Path,
    evaluation_seeds: Sequence[int],
    domains: Sequence[str],
    perfect_next_slot_csi: bool,
    resume: bool,
) -> dict[str, Any]:
    seeds = _validate_seeds(evaluation_seeds)
    selected_domains = tuple(domains)
    if not selected_domains or set(selected_domains) - {
        "in_domain",
        "out_of_domain",
    }:
        raise ValueError("domains must contain in_domain and/or out_of_domain")
    source_path = Path(run_config_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    root = Path(output_root)
    non_scanned_factor = {
        "speed": (
            "task_distributions.in_domain_test."
            "user_rician_factor_intervals_db"
        ),
        "rician": (
            "task_distributions.in_domain_test."
            "user_speed_intervals_m_per_s"
        ),
    }[scan_axis]
    point_value_key = {
        "speed": "speed_m_per_s",
        "rician": "rician_factor_db",
    }[scan_axis]
    manifest = {
        "scan_axis": scan_axis,
        "scan_values": list(scan_values),
        "evaluation_seeds": list(seeds),
        "evaluation_domains": list(selected_domains),
        "non_scanned_factor_sampling": non_scanned_factor,
        "system_config_path": str(system_config_path),
        "run_config_path": str(run_config_path),
        "unfolding_config_path": str(unfolding_config_path),
        "checkpoint_seed_directory": str(checkpoint_seed_directory),
        "perfect_next_slot_csi": perfect_next_slot_csi,
    }
    manifest_path = root / f"{scan_axis}_scan_manifest.json"
    if root.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite output: {root}")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"existing output has no manifest: {manifest_path}"
            )
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != manifest:
            raise ValueError(
                f"existing {scan_axis} scan manifest does not match"
            )
    else:
        root.mkdir(parents=True)
        _write_json(manifest_path, manifest)

    config_root = root / "generated_run_configs"
    config_root.mkdir(exist_ok=True)
    point_summaries: list[dict[str, Any]] = []
    point_summary_path = root / "point_summaries.json"
    if point_summary_path.is_file():
        point_summaries = json.loads(point_summary_path.read_text(encoding="utf-8"))

    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(exist_ok=True)
        seed_config_root = config_root / f"seed_{seed}"
        seed_config_root.mkdir(exist_ok=True)
        for scan_value in scan_values:
            point_id = _point_identifier(scan_value)
            point_key = f"seed_{seed}/{scan_axis}_{point_id}"
            if any(row["point_key"] == point_key for row in point_summaries):
                continue
            point_output = seed_root / f"{scan_axis}_{point_id}"
            if point_output.exists():
                raise FileExistsError(
                    f"partial point output already exists: {point_output}"
                )
            if scan_axis == "speed":
                point_config = build_speed_run_config(
                    source,
                    root_seed=seed,
                    speed_m_per_s=scan_value,
                )
            elif scan_axis == "rician":
                point_config = build_rician_run_config(
                    source,
                    root_seed=seed,
                    rician_factor_db=scan_value,
                )
            else:
                raise ValueError(f"unsupported scan_axis: {scan_axis}")
            point_config_path = (
                seed_config_root / f"{scan_axis}_{point_id}.json"
            )
            _write_json(point_config_path, point_config)
            summary = evaluate_padu(
                system_config_path=system_config_path,
                run_config_path=point_config_path,
                unfolding_config_path=unfolding_config_path,
                checkpoint_seed_directory=checkpoint_seed_directory,
                output_root=point_output,
                domains=selected_domains,
                perfect_next_slot_csi=perfect_next_slot_csi,
            )
            point_summaries.append(
                {
                    "point_key": point_key,
                    "root_seed": seed,
                    point_value_key: scan_value,
                    "summary": summary,
                }
            )
            _write_json(point_summary_path, point_summaries)
            _write_aggregate_outputs(
                root,
                point_summaries,
                scan_axis,
                scan_values,
                selected_domains,
            )

    _write_aggregate_outputs(
        root,
        point_summaries,
        scan_axis,
        scan_values,
        selected_domains,
    )
    return {
        "scan_axis": scan_axis,
        "scan_values": tuple(scan_values),
        "evaluation_seeds": seeds,
        "point_count": len(point_summaries),
        "output_directory": str(root),
        "aggregate_csv": str(
            root / f"{scan_axis}_scan_summary.csv"
        ),
    }


def _write_aggregate_outputs(
    root: Path,
    point_summaries: Sequence[dict[str, Any]],
    scan_axis: str,
    scan_values: Sequence[float],
    domains: Sequence[str],
) -> None:
    value_column = {
        "speed": "speed_m_per_s",
        "rician": "rician_factor_db",
    }[scan_axis]
    rows: list[dict[str, Any]] = []
    for domain in domains:
        for scan_value in scan_values:
            selected = [
                item
                for item in point_summaries
                if item[value_column] == scan_value
                and domain in item["summary"].get("domains", {})
            ]
            if not selected:
                continue
            summaries = [item["summary"]["domains"][domain] for item in selected]
            completed_slots = int(sum(item["completed_slots"] for item in summaries))
            outage_count = int(
                sum(item["joint_qos_outage_count"] for item in summaries)
            )
            throughput = [
                float(item["mean_actual_weighted_throughput_bps"])
                for item in summaries
            ]
            effective_throughput = [
                _point_effective_throughput(root, item, domain)
                for item in selected
            ]
            rows.append(
                {
                    "domain": domain,
                    value_column: scan_value,
                    "evaluation_seed_count": len(summaries),
                    "completed_slots": completed_slots,
                    "joint_qos_outage_count": outage_count,
                    "joint_qos_outage_rate": outage_count / completed_slots,
                    "mean_seed_actual_weighted_throughput_bps": mean(throughput),
                    "std_seed_actual_weighted_throughput_bps": _sample_std(throughput),
                    "pooled_actual_weighted_throughput_bps": sum(
                        float(item["mean_actual_weighted_throughput_bps"])
                        * int(item["completed_slots"])
                        for item in summaries
                    )
                    / completed_slots,
                    "mean_seed_joint_qos_effective_throughput_bps": (
                        mean(effective_throughput)
                    ),
                    "std_seed_joint_qos_effective_throughput_bps": (
                        _sample_std(effective_throughput)
                    ),
                }
            )
    _write_json(root / f"{scan_axis}_scan_summary.json", rows)
    if not rows:
        return
    with (root / f"{scan_axis}_scan_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_seeds(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError("evaluation_seeds must be non-empty and unique")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ValueError("evaluation_seeds must contain integers")
    return result


def _validate_speeds(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("speed_values must be non-empty and unique")
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("speed_values must be finite and non-negative")
    return result


def _validate_rician_factors(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("rician_values must be non-empty and unique")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("rician_values must be finite")
    return result


def _point_identifier(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def _sample_std(values: Sequence[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _point_effective_throughput(
    root: Path,
    point: dict[str, Any],
    domain: str,
) -> float:
    summary_value = point["summary"]["domains"][domain].get(
        "mean_joint_qos_effective_throughput_bps"
    )
    if summary_value is not None:
        return float(summary_value)
    slot_path = root / point["point_key"] / "slot_results.csv"
    if not slot_path.is_file():
        raise FileNotFoundError(
            f"slot result file is required for effective throughput: {slot_path}"
        )
    values: list[float] = []
    with slot_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            throughput = float(row["actual_weighted_throughput_bps"])
            outage = int(row["joint_qos_outage"])
            values.append(0.0 if outage else throughput)
    if not values:
        raise ValueError(f"slot result file has no rows: {slot_path}")
    return sum(values) / len(values)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
