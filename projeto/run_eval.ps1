# run_eval.ps1
# Executa eval_rl.py para múltiplos train_seeds e consolida os CSVs em um único arquivo.

param(
  [string]$ProjectDir = ".",
  [string]$Mode = "both",            # "pure" | "pirl" | "both"
  [int]$TrainSeedStart = 0,
  [int]$TrainSeedEnd = 4,
  [int]$EvalSeedBase = 1000,
  [int]$Episodes = 200,
  [int]$MaxSteps = 200,
  [double]$Margin = 0.02,
  [double]$MinTipDist = 0.07,
  [double]$PhiMin = -1.57079632679,  # -pi/2
  [double]$PhiMax =  1.57079632679,  # +pi/2
  [double]$SuccessTol = 0.05,
  [string]$OutDir = "results"
)

$ErrorActionPreference = "Stop"

# Resolve paths
$proj = Resolve-Path $ProjectDir
$evalScript = Join-Path $proj "eval_rl.py"
if (!(Test-Path $evalScript)) {
  throw "Não encontrei eval_rl.py em: $evalScript"
}

$outDirFull = Join-Path $proj $OutDir
New-Item -ItemType Directory -Force -Path $outDirFull | Out-Null

# Executa avaliações por seed
$csvFiles = @()

for ($s = $TrainSeedStart; $s -le $TrainSeedEnd; $s++) {
  $outCsv = Join-Path $outDirFull ("eval_{0}_train{1}_base{2}_ep{3}.csv" -f $Mode, $s, $EvalSeedBase, $Episodes)
  Write-Host "`n=== Avaliando mode=$Mode train_seed=$s (eval_seed_base=$EvalSeedBase, episodes=$Episodes) ==="

  $args = @(
    $evalScript,
    "--mode", $Mode,
    "--train-seed", $s,
    "--eval-seed-base", $EvalSeedBase,
    "--episodes", $Episodes,
    "--max-steps", $MaxSteps,
    "--margin", $Margin,
    "--min-tip-dist", $MinTipDist,
    "--phi-min", $PhiMin,
    "--phi-max", $PhiMax,
    "--success-tol", $SuccessTol,
    "--out", $outCsv
  )

  # roda python
  & python @args

  if (!(Test-Path $outCsv)) {
    throw "CSV não foi gerado: $outCsv"
  }
  $csvFiles += $outCsv
}

# Consolida CSVs (assume mesmos headers)
$merged = Join-Path $outDirFull ("eval_{0}_merged_train{1}-{2}_base{3}_ep{4}.csv" -f $Mode, $TrainSeedStart, $TrainSeedEnd, $EvalSeedBase, $Episodes)

Write-Host "`n=== Consolidando CSVs em: $merged ==="

$first = $true
$linesWritten = 0

foreach ($f in $csvFiles) {
  $lines = Get-Content -Path $f
  if ($lines.Count -lt 2) { continue }

  if ($first) {
    # escreve header + conteúdo
    Set-Content -Path $merged -Value $lines
    $linesWritten += $lines.Count
    $first = $false
  } else {
    # pula header (primeira linha)
    Add-Content -Path $merged -Value $lines[1..($lines.Count - 1)]
    $linesWritten += ($lines.Count - 1)
  }
}

Write-Host "OK! CSV consolidado: $merged"
Write-Host "Arquivos individuais:"
$csvFiles | ForEach-Object { Write-Host " - $_" }
