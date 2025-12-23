param(
  [int]$TrainSeedMin = 0,
  [int]$TrainSeedMax = 4,

  [int]$EvalSeedBase = 1000,
  [int]$Episodes = 200,

  [string]$OutDir = "results",
  [switch]$PrintEpisodes
)

$ErrorActionPreference = "Stop"

Write-Host "=== RUN EVAL GRID ==="
Write-Host "Train seeds: $TrainSeedMin..$TrainSeedMax"
Write-Host "EvalSeedBase: $EvalSeedBase | Episodes: $Episodes"
Write-Host "OutDir: $OutDir"
Write-Host "----------------------"

if (!(Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

for ($s = $TrainSeedMin; $s -le $TrainSeedMax; $s++) {
  Write-Host ""
  Write-Host "-> Evaluating train_seed=$s (exp-id=all)"
  $cmd = @(
    "python","eval_rl.py",
    "--exp-id","all",
    "--train-seed",$s,
    "--eval-seed-base",$EvalSeedBase,
    "--episodes",$Episodes,
    "--out-dir",$OutDir
  )

  if ($PrintEpisodes) {
    $cmd += "--print-episodes"
  }

  Write-Host ("CMD: " + ($cmd -join " "))
  & $cmd[0] $cmd[1..($cmd.Length-1)]
}

Write-Host ""
Write-Host "[OK] Avaliações concluídas."
Write-Host "Dica: o eval salva CSV + summary JSON por exp_id e também um summaries agregado por seed em $OutDir."
