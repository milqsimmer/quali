param(
  # Training
  [int]$Steps = 300000,
  [int]$MaxSteps = 200,
  [int[]]$Seeds = @(0,1,2,3,4),

  # Env / reward
  [double]$TauMax = 20.0,
  [double]$LamA = 0.001,
  [double]$SuccessTol = 0.05,
  [ValidateSet("dist","progress","exp")]
  [string]$TaskReward = "dist",
  [double]$ExpK = 5.0,
  [double]$LamTime = 0.0,
  [double]$LamV = 0.0,
  [double]$LamSmooth = 0.0,
  [double]$LamQ = 0.0,
  [double]$SuccessBonus = 0.0,

  # A2 sampler
  [double]$Margin = 0.02,
  [double]$MinTipDist = 0.07,
  [double]$PhiMin = -1.5707963267948966,  # -pi/2
  [double]$PhiMax =  1.5707963267948966,  # +pi/2

  # PI reward
  [double]$AlphaPi = 0.0002,
  [ValidateSet("tau_l1","power")]
  [string]$PiMetric = "tau_l1",

  # Utility
  [string]$ExpPrefix = "A0",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== RUN TRAIN (baseline + PI) ==="
Write-Host ("Seeds: " + ($Seeds -join ","))
Write-Host "Steps: $Steps | MaxSteps(ep): $MaxSteps | SuccessTol: $SuccessTol"
Write-Host "A2: margin=$Margin minTipDist=$MinTipDist phi=[$PhiMin,$PhiMax]"
Write-Host "TauMax=$TauMax lam_a=$LamA"
Write-Host "TaskReward=$TaskReward exp_k=$ExpK lt=$LamTime lv=$LamV ls=$LamSmooth lq=$LamQ sb=$SuccessBonus"
Write-Host "PI: metric=$PiMetric alpha=$AlphaPi"
Write-Host "ExpPrefix: $ExpPrefix"
Write-Host "-----------------------"

# Base args (sem PI!)
$baseArgs = @(
  "--steps", $Steps,
  "--max-steps", $MaxSteps,
  "--seed", 0,  # placeholder
  "--tau-max", $TauMax,
  "--lam-a", $LamA,
  "--success-tol", $SuccessTol,

  "--task-reward", $TaskReward,
  "--exp-k", $ExpK,
  "--lam-time", $LamTime,
  "--lam-v", $LamV,
  "--lam-smooth", $LamSmooth,
  "--lam-q", $LamQ,
  "--success-bonus", $SuccessBonus,

  "--margin", $Margin,
  "--min-tip-dist", $MinTipDist,
  "--phi-min", $PhiMin,
  "--phi-max", $PhiMax
)

# Dois modos só (baseline + PI)
$modes = @(
  @{
    Name="baseline_direct"
    Args=@(
      "--control","direct",
      "--safety-filter","none",
      "--exp-id","$ExpPrefix`__baseline_direct"
    )
  },
  @{
    Name="pi_direct"
    Args=@(
      "--control","direct",
      "--use-pi-reward",
      "--safety-filter","none",
      "--pi-metric",$PiMetric,
      "--alpha-pi",$AlphaPi,
      "--exp-id","$ExpPrefix`__pi_direct__$PiMetric`_a$AlphaPi"
    )
  }
)

foreach ($m in $modes) {
  Write-Host ""
  Write-Host "=== Mode: $($m.Name) ==="

  foreach ($s in $Seeds) {
    Write-Host ""
    Write-Host "-> Training seed=$s"

    $cmd = @("python","train_rl.py") + $baseArgs + $m.Args

    # troca seed
    for ($i=0; $i -lt $cmd.Length; $i++) {
      if ($cmd[$i] -eq "--seed") { $cmd[$i+1] = $s; break }
    }

    Write-Host ("CMD: " + ($cmd -join " "))

    if (-not $DryRun) {
      & $cmd[0] $cmd[1..($cmd.Length-1)]
    }
  }
}

Write-Host ""
Write-Host "[OK] Treinos concluídos."
