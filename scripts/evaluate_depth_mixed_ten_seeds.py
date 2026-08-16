"""Evaluate PADU depth ablations on common Mixed trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from padu.unfolding_workflow import evaluate_padu
except (ImportError, OSError, ValueError) as error:
    raise SystemExit(
        "PADU dependencies could not be loaded in the current Python "
        "environment. Run this script in the tested drl environment."
    ) from error


DEFAULT_SEEDS = (101, 201, 301, 401, 501, 601, 701, 801, 901, 1001)
DEFAULT_DEPTHS = (1, 2, 3, 4)
REQUIRED_RESULT_FILES = (
    "slot_results.csv",
    "summary.json",
    "seeds.json",
    "task_parameters.json",
)


@dataclass(frozen=True)
class DepthSpec:
    depth: int
    unfolding_config: str
    checkpoint_directory: str


DEPTH_SPECS = {
    1: DepthSpec(
        depth=1,
        unfolding_config="configs/unfolding.padu_l1.json",
        checkpoint_directory="artifacts/checkpoints/padu_l1",
    ),
    2: DepthSpec(
        depth=2,
        unfolding_config="configs/unfolding.padu_l2.json",
        checkpoint_directory="artifacts/checkpoints/padu_l2",
    ),
    3: DepthSpec(
        depth=3,
        unfolding_config="configs/unfolding.padu_l3.json",
        checkpoint_directory="artifacts/checkpoints/padu_l3",
    ),
    4: DepthSpec(
        depth=4,
        unfolding_config="configs/unfolding.padu_l4.json",
        checkpoint_directory="artifacts/checkpoints/padu_l4",
    ),
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PADU depth ablations over common Mixed trajectories."
    )
    parser.add_argument("--system", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        choices=tuple(DEFAULT_DEPTHS),
        default=list(DEFAULT_DEPTHS),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=("in_domain", "out_of_domain"),
        default=("in_domain",),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_run_config(source: dict[str, Any], root_seed: int) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    result["root_seeds"] = [root_seed]
    return result


def _validate_depth_spec(spec: DepthSpec) -> None:
    """Check that each selected depth has a matching config and checkpoint."""
    unfolding_path = ROOT / spec.unfolding_config
    if not unfolding_path.is_file():
        raise FileNotFoundError(f"missing unfolding config: {unfolding_path}")
    settings = json.loads(unfolding_path.read_text(encoding="utf-8"))
    architecture = settings.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(
            f"unfolding config has no architecture object: {unfolding_path}"
        )
    configured_depth = architecture.get("unfolding_layers")
    if configured_depth != spec.depth:
        raise ValueError(
            f"unfolding depth mismatch for L={spec.depth}: "
            f"{unfolding_path} declares {configured_depth}"
        )

    checkpoint_root = ROOT / spec.checkpoint_directory
    required_files = (
        checkpoint_root / "seeds.json",
        checkpoint_root / "normalizer.npz",
        checkpoint_root / "task_parameters.json",
        checkpoint_root / "checkpoints" / "padu_controller.pt",
        checkpoint_root / "checkpoints" / "predictor.pt",
    )
    missing = [
        str(path.relative_to(ROOT))
        for path in required_files
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"checkpoint for L={spec.depth} is incomplete: {missing}"
        )


def _is_true(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"invalid boolean value in slot_results.csv: {value}")


def _load_slot_rows(path: Path, domains: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty slot result file: {path}")
        required = {
            "root_seed",
            "domain",
            "trajectory_id",
            "target_slot",
            "actual_weighted_throughput_bps",
            "joint_qos_outage",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [
            row for row in reader
            if row["domain"] in domains
        ]
    if not rows:
        raise ValueError(f"no selected-domain rows in {path}")
    return rows


def _slot_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        row["domain"],
        row["trajectory_id"],
        int(row["target_slot"]),
    )


def _summarize_rows(
    rows: list[dict[str, str]],
    *,
    depth: int,
    root_seed: int | None,
) -> dict[str, Any]:
    outage_count = sum(_is_true(row["joint_qos_outage"]) for row in rows)
    total_slots = len(rows)
    throughput_bps = [
        float(row["actual_weighted_throughput_bps"])
        for row in rows
    ]
    qos_effective_bps = [
        0.0 if _is_true(row["joint_qos_outage"]) else value
        for row, value in zip(rows, throughput_bps)
    ]
    result: dict[str, Any] = {
        "L": depth,
        "outage_count": outage_count,
        "total_valid_slots": total_slots,
        "outage_rate_percent": 100.0 * outage_count / total_slots,
        "average_throughput_mbps": sum(throughput_bps) / total_slots / 1e6,
        "qos_effective_throughput_mbps": (
            sum(qos_effective_bps) / total_slots / 1e6
        ),
    }
    if root_seed is not None:
        result = {"root_seed": root_seed, **result}
    return result


def _evaluate_one(
    *,
    system: Path,
    run_config: Path,
    spec: DepthSpec,
    output: Path,
    domains: tuple[str, ...],
    resume: bool,
) -> None:
    if output.exists():
        complete = all(
            (output / filename).is_file()
            for filename in REQUIRED_RESULT_FILES
        )
        if resume and complete:
            return
        if resume and not any(output.iterdir()):
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"refusing to reuse incomplete or existing output: {output}"
            )
    evaluate_padu(
        system_config_path=system,
        run_config_path=run_config,
        unfolding_config_path=ROOT / spec.unfolding_config,
        checkpoint_seed_directory=ROOT / spec.checkpoint_directory,
        output_root=output,
        domains=domains,
    )


def _audit_pairing(
    *,
    output: Path,
    depths: tuple[int, ...],
    seeds: tuple[int, ...],
    domains: tuple[str, ...],
) -> dict[str, Any]:
    seed_audits: list[dict[str, Any]] = []
    all_passed = True
    for root_seed in seeds:
        directories = {
            depth: output / f"L{depth}" / f"seed_{root_seed}"
            for depth in depths
        }
        reference = directories[depths[0]]
        reference_seed_hash = _sha256(reference / "seeds.json")
        reference_task_hash = _sha256(reference / "task_parameters.json")
        reference_rows = _load_slot_rows(
            reference / "slot_results.csv",
            domains,
        )
        reference_keys = {_slot_key(row) for row in reference_rows}
        seed_audit: dict[str, Any] = {
            "root_seed": root_seed,
            "shared_seeds_sha256": reference_seed_hash,
            "shared_task_parameters_sha256": reference_task_hash,
            "depths": {},
        }
        for depth, directory in directories.items():
            rows = _load_slot_rows(directory / "slot_results.csv", domains)
            keys = [_slot_key(row) for row in rows]
            no_duplicate_keys = len(keys) == len(set(keys))
            depth_ok = {
                "rows": len(rows),
                "seeds_sha256_matches": (
                    _sha256(directory / "seeds.json")
                    == reference_seed_hash
                ),
                "task_parameters_sha256_matches": (
                    _sha256(directory / "task_parameters.json")
                    == reference_task_hash
                ),
                "slot_keys_match": set(keys) == reference_keys,
                "no_duplicate_slot_keys": no_duplicate_keys,
            }
            depth_ok["passed"] = all(depth_ok.values())
            seed_audit["depths"][str(depth)] = depth_ok
            all_passed = all_passed and depth_ok["passed"]
        seed_audits.append(seed_audit)
    return {
        "schema_version": 1,
        "all_passed": all_passed,
        "seeds": list(seeds),
        "depths": list(depths),
        "per_seed": seed_audits,
    }


def main() -> None:
    arguments = _parse_arguments()
    seeds = tuple(dict.fromkeys(arguments.seeds))
    depths = tuple(dict.fromkeys(arguments.depths))
    domains = tuple(dict.fromkeys(arguments.domains))
    if not seeds:
        raise ValueError("--seeds must not be empty")
    if not depths:
        raise ValueError("--depths must not be empty")
    if not domains:
        raise ValueError("--domains must not be empty")

    source_run = json.loads(arguments.run.read_text(encoding="utf-8"))
    if not isinstance(source_run, dict):
        raise ValueError("--run must contain a JSON object")
    for depth in depths:
        _validate_depth_spec(DEPTH_SPECS[depth])

    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "system": str(arguments.system),
        "run": str(arguments.run),
        "seeds": list(seeds),
        "depths": list(depths),
        "domains": list(domains),
    }
    manifest_path = output / "depth_ablation_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest != manifest:
            raise ValueError(
                "existing depth ablation manifest does not match"
            )
    else:
        _write_json(manifest_path, manifest)

    generated_configs = output / "generated_run_configs"
    per_seed_rows: list[dict[str, Any]] = []
    pooled_rows: dict[int, list[dict[str, str]]] = {
        depth: [] for depth in depths
    }
    for depth in depths:
        spec = DEPTH_SPECS[depth]
        for root_seed in seeds:
            run_config = _build_run_config(source_run, root_seed)
            run_path = generated_configs / f"seed_{root_seed}.json"
            _write_json(run_path, run_config)
            seed_output = output / f"L{depth}" / f"seed_{root_seed}"
            _evaluate_one(
                system=arguments.system,
                run_config=run_path,
                spec=spec,
                output=seed_output,
                domains=domains,
                resume=arguments.resume,
            )
            rows = _load_slot_rows(
                seed_output / "slot_results.csv",
                domains,
            )
            pooled_rows[depth].extend(rows)
            per_seed_rows.append(
                _summarize_rows(
                    rows,
                    depth=depth,
                    root_seed=root_seed,
                )
            )

    per_seed_rows.sort(key=lambda row: (row["root_seed"], row["L"]))
    _write_csv(output / "per_seed_summary.csv", per_seed_rows)
    pooled_summary = [
        _summarize_rows(pooled_rows[depth], depth=depth, root_seed=None)
        for depth in depths
    ]
    pooled_summary.sort(key=lambda row: row["L"])
    _write_csv(output / "pooled_summary.csv", pooled_summary)
    _write_json(
        output / "pairing_audit.json",
        _audit_pairing(
            output=output,
            depths=depths,
            seeds=seeds,
            domains=domains,
        ),
    )
    print(
        json.dumps(
            {
                "output_directory": str(output),
                "seeds": list(seeds),
                "depths": list(depths),
                "domains": list(domains),
                "per_seed_summary": str(output / "per_seed_summary.csv"),
                "pooled_summary": str(output / "pooled_summary.csv"),
                "pairing_audit": str(output / "pairing_audit.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
