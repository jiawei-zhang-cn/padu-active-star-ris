"""Command-line entry points for PADU experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="padu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--system", required=True)
    audit.add_argument("--run", required=True)

    train_unfolding = subparsers.add_parser("train")
    train_unfolding.add_argument("--system", required=True)
    train_unfolding.add_argument("--run", required=True)
    train_unfolding.add_argument("--unfolding", required=True)
    train_unfolding.add_argument(
        "--pretrained-predictor-directory",
        help=(
            "Existing PADU training output whose predictor is reused "
            "while training a new controller"
        ),
    )
    train_unfolding.add_argument("--output", required=True)

    evaluate_unfolding = subparsers.add_parser("evaluate")
    evaluate_unfolding.add_argument("--system", required=True)
    evaluate_unfolding.add_argument("--run", required=True)
    evaluate_unfolding.add_argument("--unfolding", required=True)
    evaluate_unfolding.add_argument(
        "--checkpoint-seed-directory", required=True
    )
    evaluate_unfolding.add_argument(
        "--domains",
        nargs="+",
        choices=("in_domain", "out_of_domain"),
        default=("in_domain", "out_of_domain"),
    )
    evaluate_unfolding.add_argument(
        "--perfect-next-slot-csi",
        action="store_true",
        help=(
            "Replace the controller channel input and unfolding scenarios "
            "with the exact next-slot CSI during evaluation"
        ),
    )
    evaluate_unfolding.add_argument("--output", required=True)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--system", required=True)
    scan.add_argument("--run", required=True)
    scan.add_argument("--unfolding", required=True)
    scan.add_argument(
        "--checkpoint-seed-directory", required=True
    )
    scan.add_argument(
        "--scan-axis",
        choices=("speed", "rician"),
        default="speed",
    )
    scan.add_argument("--seeds", nargs="+", type=int, required=True)
    scan.add_argument("--values", nargs="+", type=float)
    scan.add_argument(
        "--domains",
        nargs="+",
        choices=("in_domain", "out_of_domain"),
        default=("in_domain",),
    )
    scan.add_argument(
        "--perfect-next-slot-csi",
        action="store_true",
    )
    scan.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    scan.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "audit":
        from .experiment import load_configs

        system, run, audit = load_configs(arguments.system, arguments.run)
        print(
            json.dumps(
                {
                    "number_of_users": system.number_of_users,
                    "number_of_surface_elements": (
                        system.array.star_elements
                    ),
                    "root_seeds": list(run.root_seeds),
                    "audit": asdict(audit),
                },
                indent=2,
            )
        )
        return 0
    if arguments.command == "train":
        from .unfolding_workflow import train_padu

        summary = train_padu(
            system_config_path=arguments.system,
            run_config_path=arguments.run,
            unfolding_config_path=arguments.unfolding,
            pretrained_predictor_directory=(
                arguments.pretrained_predictor_directory
            ),
            output_root=arguments.output,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if arguments.command == "evaluate":
        from .unfolding_workflow import evaluate_padu

        summary = evaluate_padu(
            system_config_path=arguments.system,
            run_config_path=arguments.run,
            unfolding_config_path=arguments.unfolding,
            checkpoint_seed_directory=(
                arguments.checkpoint_seed_directory
            ),
            output_root=arguments.output,
            domains=arguments.domains,
            perfect_next_slot_csi=arguments.perfect_next_slot_csi,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if arguments.command == "scan":
        from .scan_workflow import (
            run_rician_scan,
            run_speed_scan,
        )

        common_arguments = {
            "system_config_path": arguments.system,
            "run_config_path": arguments.run,
            "unfolding_config_path": arguments.unfolding,
            "checkpoint_seed_directory": arguments.checkpoint_seed_directory,
            "output_root": arguments.output,
            "evaluation_seeds": arguments.seeds,
            "domains": arguments.domains,
            "perfect_next_slot_csi": arguments.perfect_next_slot_csi,
            "resume": arguments.resume,
        }
        if arguments.scan_axis == "speed":
            summary = run_speed_scan(
                **common_arguments,
                speed_values=arguments.values,
            )
        else:
            if arguments.values is None:
                raise ValueError("--values is required for a Rician scan")
            summary = run_rician_scan(
                **common_arguments,
                rician_values=arguments.values,
            )
        print(json.dumps(summary, indent=2))
        return 0
    raise RuntimeError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
