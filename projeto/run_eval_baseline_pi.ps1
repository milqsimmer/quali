# run_eval_baseline_pi.ps1

# run: .\run_eval_baseline_pi.ps1 -ExpPrefix A0p -TrainSeeds (0..4) -PiExpId "A0p__pi_power_a2e-5__gate_d0p25_dm0p05"


param(
  # Treinos a avaliar
  [int[]]$TrainSeeds = @(),
  [int]$TrainSeedMin = 0,
  [int]$TrainSeedMax = 4,

  # IDs
  [string]$ExpPrefix = "A0",
  [ValidateSet("tau_l1","power")]
  [string]$PiMetric = "tau_l1",
  [double]$AlphaPi = 0.0002,

  # Override opcional pro PI (quando exp-id é custom, ex.: com gating no nome)
  [string]$PiExpId = "",

  # Avaliação
  [int]$EvalSeedBase = 1000,
  [int]$Episodes = 200,

  # Saída
  [string]$OutDir = "results",

  # Flags
  [switch]$Render,
  [switch]$PrintEpisodes,
  [switch]$Strict,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($TrainSeeds.Count -eq 0) {
  $TrainSeeds = @($TrainSeedMin..$TrainSeedMax)
}

# evita notação científica no alpha
$alphaStr = "{0:0.################}" -f $AlphaPi

$baselineExpId = "$ExpPrefix`__baseline_direct"
$defaultPiExpId = "$ExpPrefix`__pi_direct__$PiMetric`_a$alphaStr"
$piExpId = if ($PiExpId -ne "") { $PiExpId } else { $defaultPiExpId }

$modes = @(
  @{ Name="baseline"; ExpId=$baselineExpId },
  @{ Name="pi";       ExpId=$piExpId }
)

Write-Host "=== RUN EVAL (baseline + PI only) ==="
Write-Host ("Train seeds: " + ($TrainSeeds -join ", "))
Write-Host "Baseline ExpId: $baselineExpId"
Write-Host "PI ExpId:       $piExpId"
Write-Host "EvalSeedBase: $EvalSeedBase | Episodes: $Episodes"
Write-Host "OutDir: $OutDir"
Write-Host "Strict: $Strict"
Write-Host "----------------------"

if (!(Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

foreach ($m in $modes) {
  Write-Host ""
  Write-Host "=== Mode: $($m.Name) | exp-id=$($m.ExpId) ==="

  foreach ($s in $TrainSeeds) {
    Write-Host ""
    Write-Host "-> Evaluating train_seed=$s"

    $cmd = @(
      "python","eval_rl.py",
      "--exp-id",$m.ExpId,
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
Write-Host "Arquivos: CSV + summary JSON por exp_id em $OutDir."
Write-Host "Dica: compare $baselineExpId vs $piExpId (mesmas seeds/episódios)."
