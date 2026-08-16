"""Evaluate and pool the submitted PADU methods on common Mixed trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from padu.unfolding_workflow import evaluate_padu
except (ImportError, OSError, ValueError) as error:
    raise SystemExit(
        "PADU dependencies could not be loaded in the current Python "
        "environment. Run this script in the tested drl environment, "
        "for example: conda run --no-capture-output -n drl python "
        "scripts/evaluate_mixed_ten_seeds.py ..."
    ) from error


DEFAULT_SEEDS = (101, 201, 301, 401, 501, 601, 701, 801, 901, 1001)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    unfolding_config: str
    checkpoint_directory: str
    perfect_next_slot_csi: bool = False


METHODS = (
    MethodSpec(
        key="padu",
        unfolding_config="configs/unfolding.padu_l3.json",
        checkpoint_directory="artifacts/checkpoints/padu_l3",
    ),
    MethodSpec(
        key="benchmark1_without_csi_prediction",
        unfolding_config=(
            "configs/unfolding.benchmark1_without_csi_prediction.json"
        ),
        checkpoint_directory=(
            "artifacts/checkpoints/benchmark1_without_csi_prediction"
        ),
    ),
    MethodSpec(
        key="benchmark2_without_unfolding_refinement",
        unfolding_config=(
            "configs/unfolding.benchmark2_without_unfolding_refinement.json"
        ),
        checkpoint_directory=(
            "artifacts/checkpoints/benchmark2_without_unfolding_refinement"
        ),
    ),
    MethodSpec(
        key="benchmark3_perfect_next_slot_csi",
        unfolding_config="configs/unfolding.padu_l3.json",
        checkpoint_directory="artifacts/checkpoints/padu_l3",
        perfect_next_slot_csi=True,
    ),
    MethodSpec(
        key="benchmark4_nmse_trained_point_gru",
        unfolding_config=(
            "configs/unfolding.benchmark4_nmse_trained_point_gru.json"
        ),
        checkpoint_directory=(
            "artifacts/checkpoints/benchmark4_nmse_trained_point_gru"
        ),
    ),
)

SLOT_KEY_FIELDS = ("domain", "trajectory_id", "target_slot")
SLOT_FIELDS = {
    "root_seed",
    "domain",
    "trajectory_id",
    "target_slot",
    "prediction_nmse",
    "actual_weighted_throughput_bps",
    "joint_qos_outage",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and pool PADU Mixed results over multiple seeds."
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=("in_domain", "out_of_domain"),
        default=("in_domain",),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(method.key for method in METHODS),
        default=tuple(method.key for method in METHODS),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip evaluation and summarize existing per-seed outputs.",
    )
    return parser.parse_args()


def _validate_seeds(values: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return seeds


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_slot_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing slot result file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty slot result file: {path}")
        missing = SLOT_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"slot result file has no rows: {path}")
    return rows


def _slot_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        row["domain"],
        row["trajectory_id"],
        int(row["target_slot"]),
    )


def _summary_row(
    *,
    method: str,
    root_seed: int,
    domain: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no rows for {method}, seed={root_seed}, {domain}")
    outage_count = sum(int(row["joint_qos_outage"]) for row in rows)
    throughput = [
        float(row["actual_weighted_throughput_bps"]) for row in rows
    ]
    effective = [
        0.0 if int(row["joint_qos_outage"]) else value
        for row, value in zip(rows, throughput)
    ]
    return {
        "method": method,
        "root_seed": root_seed,
        "domain": domain,
        "valid_slots": len(rows),
        "joint_qos_outage_count": outage_count,
        "joint_qos_outage_rate_percent": (
            100.0 * outage_count / len(rows)
        ),
        "mean_prediction_nmse": sum(
            float(row["prediction_nmse"]) for row in rows
        )
        / len(rows),
        "average_sum_throughput_mbit_s": (
            sum(throughput) / len(throughput) / 1.0e6
        ),
        "qos_effective_throughput_mbit_s": (
            sum(effective) / len(effective) / 1.0e6
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_run_config(
    source: dict[str, Any],
    *,
    root_seed: int,
) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    result["root_seeds"] = [root_seed]
    return result


def _evaluate(
    *,
    system: Path,
    base_run: Path,
    unfolding: Path,
    checkpoint: Path,
    output: Path,
    domains: tuple[str, ...],
    root_seed: int,
    perfect_next_slot_csi: bool,
    resume: bool,
) -> None:
    if output.exists():
        required_files = (
            "slot_results.csv",
            "summary.json",
            "seeds.json",
            "task_parameters.json",
        )
        if resume and all((output / name).is_file() for name in required_files):
            return
        if resume and not any(output.iterdir()):
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"refusing to reuse incomplete or existing output: {output}"
            )
    evaluate_padu(
        system_config_path=system,
        run_config_path=base_run,
        unfolding_config_path=unfolding,
        checkpoint_seed_directory=checkpoint,
        output_root=output,
        domains=domains,
        perfect_next_slot_csi=perfect_next_slot_csi,
    )


def _audit_pairing(
    *,
    output: Path,
    methods: tuple[MethodSpec, ...],
    seeds: tuple[int, ...],
    domains: tuple[str, ...],
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    all_passed = True
    for seed in seeds:
        seed_result: dict[str, Any] = {
            "root_seed": seed,
            "domains": {},
        }
        for domain in domains:
            method_keys: dict[str, set[tuple[str, str, int]]] = {}
            seeds_hashes: dict[str, str] = {}
            task_hashes: dict[str, str] = {}
            for method in methods:
                directory = output / method.key / f"seed_{seed}"
                rows = _load_slot_rows(directory / "slot_results.csv")
                selected = {
                    _slot_key(row) for row in rows if row["domain"] == domain
                }
                method_keys[method.key] = selected
                seeds_hashes[method.key] = _sha256(
                    directory / "seeds.json"
                )
                task_hashes[method.key] = _sha256(
                    directory / "task_parameters.json"
                )
            key_counts = {
                method: len(keys) for method, keys in method_keys.items()
            }
            same_keys = len({frozenset(keys) for keys in method_keys.values()}) == 1
            same_seeds = len(set(seeds_hashes.values())) == 1
            same_tasks = len(set(task_hashes.values())) == 1
            passed = same_keys and same_seeds and same_tasks
            all_passed = all_passed and passed
            seed_result["domains"][domain] = {
                "passed": passed,
                "same_slot_keys": same_keys,
                "same_seeds_json": same_seeds,
                "same_task_parameters_json": same_tasks,
                "slot_key_counts": key_counts,
            }
        per_seed.append(seed_result)
    result = {
        "all_passed": all_passed,
        "methods": [method.key for method in methods],
        "root_seeds": list(seeds),
        "domains": list(domains),
        "per_seed": per_seed,
    }
    _write_json(output / "pairing_audit.json", result)
    return result


def _summarize(
    *,
    output: Path,
    methods: tuple[MethodSpec, ...],
    seeds: tuple[int, ...],
    domains: tuple[str, ...],
) -> None:
    per_seed_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for method in methods:
        for domain in domains:
            pooled: list[dict[str, Any]] = []
            for seed in seeds:
                rows = _load_slot_rows(
                    output
                    / method.key
                    / f"seed_{seed}"
                    / "slot_results.csv"
                )
                selected = [row for row in rows if row["domain"] == domain]
                per_seed_rows.append(
                    _summary_row(
                        method=method.key,
                        root_seed=seed,
                        domain=domain,
                        rows=selected,
                    )
                )
                pooled.extend(selected)
            pooled_rows.append(
                _summary_row(
                    method=method.key,
                    root_seed=0,
                    domain=domain,
                    rows=pooled,
                )
                | {"root_seed": "pooled"},
            )
    _write_csv(output / "per_seed_summary.csv", per_seed_rows)
    _write_csv(output / "pooled_summary.csv", pooled_rows)
    _audit_pairing(
        output=output,
        methods=methods,
        seeds=seeds,
        domains=domains,
    )


def main() -> None:
    arguments = _parse_args()
    seeds = _validate_seeds(arguments.seeds)
    methods_by_key = {method.key: method for method in METHODS}
    methods = tuple(methods_by_key[key] for key in arguments.methods)
    if not methods:
        raise ValueError("at least one method is required")

    system = Path(arguments.system)
    base_run = Path(arguments.run)
    output = Path(arguments.output)
    source_run = json.loads(base_run.read_text(encoding="utf-8"))
    generated = output / "generated_run_configs"
    generated.mkdir(parents=True, exist_ok=True)

    if not arguments.summarize_only:
        for seed in seeds:
            run_path = generated / f"run.test_seed{seed}_mixed.json"
            _write_json(
                run_path,
                _build_run_config(source_run, root_seed=seed),
            )
            for method in methods:
                _evaluate(
                    system=system,
                    base_run=run_path,
                    unfolding=ROOT / method.unfolding_config,
                    checkpoint=ROOT / method.checkpoint_directory,
                    output=output / method.key / f"seed_{seed}",
                    domains=tuple(arguments.domains),
                    root_seed=seed,
                    perfect_next_slot_csi=method.perfect_next_slot_csi,
                    resume=arguments.resume,
                )

    _summarize(
        output=output,
        methods=methods,
        seeds=seeds,
        domains=tuple(arguments.domains),
    )
    print((output / "pooled_summary.csv").resolve())
    print((output / "per_seed_summary.csv").resolve())
    print((output / "pairing_audit.json").resolve())


if __name__ == "__main__":
    main()
