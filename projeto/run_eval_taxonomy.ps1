param(
  # Treinos a avaliar
  [int[]]$TrainSeeds = @(),
  [int]$TrainSeedMin = 0,
  [int]$TrainSeedMax = 9,

  # IDs
  [string]$ExpPrefix = "A0p",
  [ValidateSet("tau_l1","power")]
  [string]$PiMetric = "power",
  [double]$AlphaPi = 0.00002,

  [ValidateSet("proj_box","proj_box_jointlimit")]
  [string]$SafetyFilter = "proj_box_jointlimit",
  [double]$DtauMax = 2.0,
  [double]$QMargin = 0.15,

  # Avaliação
  [int]$EvalSeedBase = 1000,
  [int]$Episodes = 200,

  [string]$OutDir = "results",

  [switch]$Render,
  [switch]$PrintEpisodes,
  [switch]$Strict,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($TrainSeeds.Count -eq 0) {
  $TrainSeeds = @($TrainSeedMin..$TrainSeedMax)
}

$expIds = @(
  "$ExpPrefix`__baseline_direct",
  "$ExpPrefix`__pi_direct__${PiMetric}`_a$AlphaPi",
  "$ExpPrefix`__safety_direct__sf-${SafetyFilter}`_dt$DtauMax`_qm$QMargin",
  "$ExpPrefix`__residual_nominalPD"
)

Write-Host "=== RUN EVAL TAXONOMY (4 arquiteturas) ==="
Write-Host ("Train seeds: " + ($TrainSeeds -join ", "))
Write-Host "ExpIds:"
$expIds | ForEach-Object { Write-Host "  - $_" }
Write-Host "EvalSeedBase: $EvalSeedBase | Episodes: $Episodes"
Write-Host "OutDir: $OutDir"
Write-Host "Strict: $Strict"
Write-Host "----------------------"

if (!(Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

foreach ($expId in $expIds) {
  Write-Host ""
  Write-Host "=== exp-id=$expId ==="

  foreach ($s in $TrainSeeds) {
    Write-Host ""
    Write-Host "-> Evaluating train_seed=$s"

    $cmd = @(
      "python","eval_rl.py",
      "--exp-id",$expId,
      "--train-seed",$s,
      "--eval-seed-base",$EvalSeedBase,
      "--episodes",$Episodes,
      "--out-dir",$OutDir
    )

    if ($Render) { $cmd += "--render" }
    if ($PrintEpisodes) { $cmd += "--print-episodes" }
    if ($Strict) { $cmd += "--strict" }

    Write-Host ("CMD: " + ($cmd -join " "))

    if (-not $DryRun) {
      & $cmd[0] $cmd[1..($cmd.Length-1)]
    }
  }
}

Write-Host ""
Write-Host "[OK] Avaliações concluídas."
