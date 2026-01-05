# run_train_baseline_pi.ps1

# run: .\run_train_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00002 -PiGating distance -PiGateDist 0.25 -PiGateMinDist 0.05 -Seeds (0..4) -PiExpIdOverride "A0p__pi_power_a2e-5__gate_d0p25_dm0p05"


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

  # PI gating (distance-based)
  [ValidateSet("none","distance")]
  [string]$PiGating = "none",
  [double]$PiGateDist = 0.25,
  [double]$PiGateMinDist = 0.0,   # 0.0 => usa SuccessTol dentro do script

  # Utility
  [string]$ExpPrefix = "A0",
  [string]$PiExpIdOverride = "",  # se quiser exp-id custom do PI (ex.: A0p__pi_power_a2e-5__gate...)
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ci = [System.Globalization.CultureInfo]::InvariantCulture
function F($x) { ([double]$x).ToString("0.############################", $ci) }


# formata alpha sem notação científica (evita a2E-05)
$alphaStr = "{0:0.################}" -f $AlphaPi

# Se PiGateMinDist=0 => usa success_tol (mais conveniente)
$piGateMinStr = if ($PiGateMinDist -le 0) { $SuccessTol } else { $PiGateMinDist }

# Exp-ids
$baselineExpId = "$ExpPrefix`__baseline_direct"
$defaultPiExpId = "$ExpPrefix`__pi_direct__$PiMetric`_a$alphaStr"
$piExpId = if ($PiExpIdOverride -ne "") { $PiExpIdOverride } else { $defaultPiExpId }

Write-Host "=== RUN TRAIN (baseline + PI) ==="
Write-Host ("Seeds: " + ($Seeds -join ","))
Write-Host "Steps: $Steps | MaxSteps(ep): $MaxSteps | SuccessTol: $SuccessTol"
Write-Host "A2: margin=$Margin minTipDist=$MinTipDist phi=[$PhiMin,$PhiMax]"
Write-Host "TauMax=$TauMax lam_a(BASELINE)=$LamA | lam_a(PI)=0.0 (avoid double penalty)"
Write-Host "TaskReward=$TaskReward exp_k=$ExpK lt=$LamTime lv=$LamV ls=$LamSmooth lq=$LamQ sb=$SuccessBonus"
Write-Host "PI: metric=$PiMetric alpha=$alphaStr | gating=$PiGating gate_dist=$PiGateDist gate_min=$piGateMinStr"
Write-Host "ExpPrefix: $ExpPrefix"
Write-Host "ExpId baseline: $baselineExpId"
Write-Host "ExpId PI:       $piExpId"
Write-Host "-----------------------"

# Base args (comuns). NOTE: sem --lam-a aqui!
$baseArgs = @(
  "--steps", $Steps,
  "--max-steps", $MaxSteps,
  "--seed", 0,  # placeholder
  "--tau-max", (F $TauMax),
  "--success-tol", (F $SuccessTol),

  "--exp-k", (F $ExpK),
  "--lam-time", (F $LamTime),
  "--lam-v", (F $LamV),
  "--lam-smooth", (F $LamSmooth),
  "--lam-q", (F $LamQ),
  "--success-bonus", (F $SuccessBonus),

  "--margin", (F $Margin),
  "--min-tip-dist", (F $MinTipDist),
  "--phi-min", (F $PhiMin),
  "--phi-max", (F $PhiMax)

)

# Args de gating do PI (só aplica se use_pi_reward)
$piGateArgs = @()
if ($PiGating -ne "none") {
  $piGateArgs += @("--pi-gating", $PiGating, "--pi-gate-dist", $PiGateDist)
  if ($PiGateMinDist -gt 0) {
    $piGateArgs += @("--pi-gate-min-dist", $PiGateMinDist)
  }
  # se PiGateMinDist=0, deixamos o env usar default (success_tol) via args/manifest
}

$modes = @(
  @{
    Name="baseline_direct"
    Args=@(
      "--control","direct",
      "--safety-filter","none",
      "--lam-a", (F $LamA),
      "--exp-id",$baselineExpId
    )
  },
  @{
    Name="pi_direct"
    Args=@(
      "--control","direct",
      "--use-pi-reward",
      "--safety-filter","none",
      "--pi-metric",$PiMetric,
      "--alpha-pi", (F $AlphaPi),
      "--lam-a", "0.0",
      "--exp-id",$piExpId
    ) + $piGateArgs
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
