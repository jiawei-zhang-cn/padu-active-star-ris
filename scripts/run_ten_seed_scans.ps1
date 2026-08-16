param(
    [ValidateSet("all", "speed", "rician")]
    [string]$ScanAxis = "all"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $env:PYTHONPATH = "src"
} else {
    $env:PYTHONPATH = "src;$($env:PYTHONPATH)"
}

$EvaluationSeeds = @(101, 201, 301, 401, 501, 601, 701, 801, 901, 1001)
$SpeedValues = @("0.3", "0.6", "0.9", "1.2", "1.5", "1.8")
$RicianValues = @("0", "1", "2", "3", "4", "5", "6", "7")

$Methods = @(
    @{
        Key = "padu"
        Unfolding = "configs/unfolding.padu_l3.json"
        Checkpoint = "artifacts/checkpoints/padu_l3"
        Perfect = $false
    },
    @{
        Key = "benchmark1_without_csi_prediction"
        Unfolding = "configs/unfolding.benchmark1_without_csi_prediction.json"
        Checkpoint = "artifacts/checkpoints/benchmark1_without_csi_prediction"
        Perfect = $false
    },
    @{
        Key = "benchmark2_without_unfolding_refinement"
        Unfolding = "configs/unfolding.benchmark2_without_unfolding_refinement.json"
        Checkpoint = "artifacts/checkpoints/benchmark2_without_unfolding_refinement"
        Perfect = $false
    },
    @{
        Key = "benchmark3_perfect_next_slot_csi"
        Unfolding = "configs/unfolding.padu_l3.json"
        Checkpoint = "artifacts/checkpoints/padu_l3"
        Perfect = $true
    },
    @{
        Key = "benchmark4_nmse_trained_point_gru"
        Unfolding = "configs/unfolding.benchmark4_nmse_trained_point_gru.json"
        Checkpoint = "artifacts/checkpoints/benchmark4_nmse_trained_point_gru"
        Perfect = $false
    }
)

$Scans = @(
    @{
        Axis = "speed"
        Values = $SpeedValues
    },
    @{
        Axis = "rician"
        Values = $RicianValues
    }
)

foreach ($scan in $Scans) {
    if ($ScanAxis -ne "all" -and $scan.Axis -ne $ScanAxis) {
        continue
    }
    foreach ($method in $Methods) {
        $arguments = @(
            "scan",
            "--system", "configs/system.padu.json",
            "--run", "configs/run.test_seed101_mixed.json",
            "--unfolding", $method.Unfolding,
            "--checkpoint-seed-directory", $method.Checkpoint,
            "--scan-axis", $scan.Axis,
            "--seeds"
        ) + $EvaluationSeeds + @(
            "--values"
        ) + $scan.Values + @(
            "--domains", "in_domain",
            "--output",
            ("results/scans_10seed/{0}/{1}" -f $scan.Axis, $method.Key)
        )

        if ($method.Perfect) {
            $arguments += "--perfect-next-slot-csi"
        }

        Write-Host ("Running {0} / {1}" -f $scan.Axis, $method.Key)
        & python -m padu.cli @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "padu scan failed for $($scan.Axis) / $($method.Key)"
        }
    }
}
