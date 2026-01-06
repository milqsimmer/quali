param(
  # Training
  [int]$Steps = 300000,
  [int]$MaxSteps = 200,
  [int[]]$Seeds = @(0,1,2,3,4,5,6,7,8,9),

  # Env / reward (task)
  [double]$TauMax = 20.0,
  [double]$LamA = 0.001,          # regularização padrão (baseline / safety-only / residual)
  [double]$SuccessTol = 0.05,
  [ValidateSet("dist","progress","exp")]
  [string]$TaskReward = "exp",
  [double]$ExpK = 10.0,
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

  # PI-reward
  [ValidateSet("tau_l1","power")]
  [string]$PiMetric = "power",
  [double]$AlphaPi = 0.00002,     # 2e-5 (candidato atual)

  # Safety filter
  [ValidateSet("proj_box","proj_box_jointlimit")]
  [string]$SafetyFilter = "proj_box_jointlimit",
  [double]$DtauMax = 2.0,
  [double]$QMargin = 0.15,

  # Utility
  [string]$ExpPrefix = "A0p",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== RUN TRAIN TAXONOMY (Banerjee et al., 2023) ==="
Write-Host ("Seeds: " + ($Seeds -join ","))
Write-Host "TaskReward=$TaskReward | exp_k=$ExpK | MaxSteps=$MaxSteps | Steps=$Steps"
Write-Host "TauMax=$TauMax | SuccessTol=$SuccessTol"
Write-Host "A2: margin=$Margin minTipDist=$MinTipDist phi=[$PhiMin,$PhiMax]"
Write-Host "PI: metric=$PiMetric alpha=$AlphaPi  (lam_a será forçado a 0 no train_rl.py quando PI estiver ativo)"
Write-Host "Safety: sf=$SafetyFilter dtau_max=$DtauMax q_margin=$QMargin"
Write-Host "ExpPrefix=$ExpPrefix"
Write-Host "-----------------------"

# args comuns
$baseArgs = @(
  "--steps", $Steps,
  "--max-steps", $MaxSteps,
  "--seed", 0,  # placeholder
  "--tau-max", $TauMax,
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

# 4 arquiteturas: baseline, PI-reward, safety-filter, residual RL
$modes = @(
  @{
    Name = "baseline_rl_puro"
    Args = @(
      "--control","direct",
      "--safety-filter","none",
      "--lam-a",$LamA,
      "--exp-id","$ExpPrefix`__baseline_direct"
    )
  },
  @{
    Name = "pi_reward"
    Args = @(
      "--control","direct",
      "--use-pi-reward",
      "--pi-metric",$PiMetric,
      "--alpha-pi",$AlphaPi,
      "--safety-filter","none",
      "--lam-a",$LamA,  # será zerado automaticamente no train_rl.py (guardrail)
      "--exp-id","$ExpPrefix`__pi_direct__${PiMetric}`_a$AlphaPi"
    )
  },
  @{
    Name = "action_regulation_safety_filter"
    Args = @(
      "--control","direct",
      "--safety-filter",$SafetyFilter,
      "--dtau-max",$DtauMax,
      "--q-margin",$QMargin,
      "--lam-a",$LamA,
      "--exp-id","$ExpPrefix`__safety_direct__sf-${SafetyFilter}`_dt$DtauMax`_qm$QMargin"
    )
  },
  @{
    Name = "residual_rl"
    Args = @(
      "--control","residual",
      "--safety-filter","none",
      "--lam-a",$LamA,
      "--exp-id","$ExpPrefix`__residual_nominalPD"
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
