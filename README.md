# PADU: Prediction-Aware Deep Unfolding for Active STAR-RIS

This repository contains the implementation and evaluation code for
prediction-aware deep unfolding (PADU) in active STAR-RIS-aided
short-packet transmission. It includes the submitted PADU method,
Benchmarks 1--4, depth-ablation configurations, and the scripts used to
generate the reported figures.

## Contents

- `src/padu/`: channel generation, finite-blocklength physical model,
  probabilistic/point GRU prediction, feasible hardware mapping, and PADU
  training/evaluation code.
- `configs/`: exact JSON configurations used for the submitted experiments.
- `artifacts/checkpoints/`: trained models used for direct evaluation.
- `results/mixed_10seed/`: frozen ten-seed mixed-scenario summary and pairing audit.
- `results/depth_ablation_10seed/`: frozen ten-seed PADU depth-ablation summaries.
- `results/scans_10seed/`: frozen ten-seed speed and Rician-factor scan CSV files.
- `results/figures_10seed/`: the four submitted ten-seed PDF figures.
- `scripts/plot_scan_results.py`: regenerates the speed and
  Rician-factor PDF figures from the summary CSV files.

## Environment

The original experiments were run with Python 3.10 and CUDA-enabled PyTorch.
Install from this directory with:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

The run files set `"require_cuda": true`.  To evaluate on CPU, change this
field to `false` in a copied run configuration.

Tested environment:

```text
Python 3.10.20
PyTorch 2.10.0+cu126
CUDA 12.6
NumPy 2.2.6
SciPy 1.15.3
```

The same information is also recorded in `environment.txt`.

The commands below invoke Python directly after installation. The scan
wrapper is provided in PowerShell syntax for Windows; on Linux/macOS, run
the equivalent `python -m padu.cli` command with the same arguments.

## Main Configurations

System:

```text
configs/system.padu.json
```

Training and mixed-test run files:

```text
configs/run.train_seed1_mixed.json
configs/run.test_seed101_mixed.json
configs/run.test_seed201_mixed.json
configs/run.test_seed301_mixed.json
```

Controller configurations:

```text
configs/unfolding.padu_l3.json
configs/unfolding.benchmark1_without_csi_prediction.json
configs/unfolding.benchmark2_without_unfolding_refinement.json
configs/unfolding.benchmark4_nmse_trained_point_gru.json
configs/unfolding.padu_l1.json
configs/unfolding.padu_l2.json
configs/unfolding.padu_l4.json
```

## Method Mapping

- `PADU`: `artifacts/checkpoints/padu_l3` with
  `configs/unfolding.padu_l3.json`.
- `Benchmark 1`: `artifacts/checkpoints/benchmark1_without_csi_prediction`
  with `configs/unfolding.benchmark1_without_csi_prediction.json`.
- `Benchmark 2`: `artifacts/checkpoints/benchmark2_without_unfolding_refinement`
  with `configs/unfolding.benchmark2_without_unfolding_refinement.json`.
- `Benchmark 3`: the PADU checkpoint with `--perfect-next-slot-csi`.
- `Benchmark 4`: `artifacts/checkpoints/benchmark4_nmse_trained_point_gru`
  with `configs/unfolding.benchmark4_nmse_trained_point_gru.json`.

Legacy implementation identifiers containing `dual` or `primal_dual`
refer to the nonnegative service-shortfall weights used by PADU. They
should not be interpreted as Lagrange multipliers or as a primal-dual
convergence claim.

The submitted PADU configuration sets
`probabilistic_uncertainty_conditioning=false`. The probabilistic GRU
scale is used for offline negative-log-likelihood training only; online
control uses the predicted conditional mean and the GRU state.

## Direct Mixed Evaluation

The single-seed command below is provided as a quick evaluation example.
The mixed-scenario results reported in the paper are pooled over ten root
seeds `101, 201, 301, 401, 501, 601, 701, 801, 901, 1001`, for a total of
1920 valid test slots per method.

Example for PADU on root seed 101:

```bash
padu evaluate --system configs/system.padu.json --run configs/run.test_seed101_mixed.json --unfolding configs/unfolding.padu_l3.json --checkpoint-seed-directory artifacts/checkpoints/padu_l3 --domains in_domain --output results/reproduced_mixed_seed101_padu
```

Perfect next-slot CSI reference:

```bash
padu evaluate --system configs/system.padu.json --run configs/run.test_seed101_mixed.json --unfolding configs/unfolding.padu_l3.json --checkpoint-seed-directory artifacts/checkpoints/padu_l3 --domains in_domain --perfect-next-slot-csi --output results/reproduced_mixed_seed101_perfect_csi
```

## Training From Scratch

The provided checkpoints are the frozen models used for the reported
results. Retraining is optional and may not reproduce bitwise-identical
model parameters because of hardware and software differences.

PADU L=3 training command:

```bash
padu train --system configs/system.padu.json --run configs/run.train_seed1_mixed.json --unfolding configs/unfolding.padu_l3.json --output results/retrained_padu_l3
```

## Speed And Rician Scans

PADU speed scan:

```bash
padu scan --system configs/system.padu.json --run configs/run.test_seed101_mixed.json --unfolding configs/unfolding.padu_l3.json --checkpoint-seed-directory artifacts/checkpoints/padu_l3 --scan-axis speed --seeds 101 201 301 401 501 601 701 801 901 1001 --values 0.3 0.6 0.9 1.2 1.5 1.8 --domains in_domain --output results/reproduced_speed_padu
```

PADU Rician-factor scan:

```bash
padu scan --system configs/system.padu.json --run configs/run.test_seed101_mixed.json --unfolding configs/unfolding.padu_l3.json --checkpoint-seed-directory artifacts/checkpoints/padu_l3 --scan-axis rician --seeds 101 201 301 401 501 601 701 801 901 1001 --values 0 1 2 3 4 5 6 7 --domains in_domain --output results/reproduced_rician_padu
```

Use the checkpoint/config mapping above to run the four benchmarks.

## Ten-Seed Evaluation

The submitted paper tables and figures use ten root seeds:

```text
101, 201, 301, 401, 501, 601, 701, 801, 901, 1001
```

Mixed-scenario pooled evaluation for PADU and Benchmarks 1--4:

```bash
python scripts/evaluate_mixed_ten_seeds.py --system configs/system.padu.json --run configs/run.test_seed101_mixed.json --output results/mixed_10seed
```

The script writes:

```text
results/mixed_10seed/per_seed_summary.csv
results/mixed_10seed/pooled_summary.csv
results/mixed_10seed/pairing_audit.json
```

Ten-seed speed and Rician scans for PADU and Benchmarks 1--4:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ten_seed_scans.ps1
```

The scan wrapper launches the repository source with
`python -m padu.cli` and sets `PYTHONPATH=src`; the `padu` console script
does not need to be installed in the active environment.

To run only one scan axis:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ten_seed_scans.ps1 -ScanAxis speed
powershell -ExecutionPolicy Bypass -File scripts/run_ten_seed_scans.ps1 -ScanAxis rician
```

The scan outputs are written under:

```text
results/scans_10seed/speed/
results/scans_10seed/rician/
```

## Plot Figures

To plot the submitted ten-seed scan outputs after running
`scripts/run_ten_seed_scans.ps1`, use:

```bash
python scripts/plot_scan_results.py --scan-axis speed --scan-root results/scans_10seed/speed --output-directory results/figures_10seed --expected-seed-count 10
python scripts/plot_scan_results.py --scan-axis rician --scan-root results/scans_10seed/rician --output-directory results/figures_10seed --expected-seed-count 10
```

For the ten-seed commands above, the PDFs are written to:

```text
results/figures_10seed/
```

The submitted scan CSV files used by the plotting script are:

```text
results/scans_10seed/speed/
results/scans_10seed/rician/
```

## Depth Ablation

The official depth comparison uses trained controllers in:

```text
artifacts/checkpoints/padu_l1
artifacts/checkpoints/padu_l2
artifacts/checkpoints/padu_l3
artifacts/checkpoints/padu_l4
```

The submitted pooled summary is:

```text
results/depth_ablation_10seed/pooled_summary.csv
```

Ten-seed Mixed evaluation of the four trained PADU depths:

```bash
python scripts/evaluate_depth_mixed_ten_seeds.py --system configs/system.padu.json --run configs/run.test_seed101_mixed.json --output results/depth_ablation_10seed
```

This evaluates `L=1,2,3,4` on root seeds
`101,201,301,401,501,601,701,801,901,1001`. The script only generates
per-seed copies of the Mixed test run file; it does not train or modify any
checkpoint. Each depth uses its own trained controller and the same Mixed
trajectory protocol. Complete existing seed outputs are reused, while
incomplete non-empty outputs are retained and reported instead of being
overwritten.

The output files are:

```text
results/depth_ablation_10seed/per_seed_summary.csv
results/depth_ablation_10seed/pooled_summary.csv
results/depth_ablation_10seed/pairing_audit.json
```

`pooled_summary.csv` recomputes outage and throughput directly over all valid
slots. `pairing_audit.json` verifies `seeds.json`,
`task_parameters.json`, and target-slot keys across depths for every root
seed.
