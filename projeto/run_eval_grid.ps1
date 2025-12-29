param(
  # Escolha UMA forma:
  [int[]]$TrainSeeds = @(),
  [int]$TrainSeedMin = 0,
  [int]$TrainSeedMax = 4,

  [string]$ExpId = "all",

  [int]$EvalSeedBase = 1000,
  [int]$Episodes = 200,

  [string]$OutDir = "results",

  [switch]$Render,
  [switch]$PrintEpisodes
)

$ErrorActionPreference = "Stop"

if ($TrainSeeds.Count -eq 0) {
  $TrainSeeds = @($TrainSeedMin..$TrainSeedMax)
}

Write-Host "=== RUN EVAL GRID ==="
Write-Host ("Train seeds: " + ($TrainSeeds -join ", "))
Write-Host "ExpId: $ExpId"
Write-Host "EvalSeedBase: $EvalSeedBase | Episodes: $Episodes"
Write-Host "OutDir: $OutDir"
Write-Host "----------------------"

if (!(Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

foreach ($s in $TrainSeeds) {
  Write-Host ""
  Write-Host "-> Evaluating train_seed=$s (exp-id=$ExpId)"

  $cmd = @(
    "python","eval_rl.py",
    "--exp-id",$ExpId,
    "--train-seed",$s,
    "--eval-seed-base",$EvalSeedBase,
    "--episodes",$Episodes,
    "--out-dir",$OutDir
  )

  if ($Render) { $cmd += "--render" }
  if ($PrintEpisodes) { $cmd += "--print-episodes" }

  Write-Host ("CMD: " + ($cmd -join " "))
  & $cmd[0] $cmd[1..($cmd.Length-1)]
}

Write-Host ""
Write-Host "[OK] Avaliações concluídas."
Write-Host "Dica: o eval salva CSV + summary JSON por exp_id e também um summaries agregado por seed em $OutDir."
