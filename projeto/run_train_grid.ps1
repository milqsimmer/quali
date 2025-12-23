param(
  [int]$Steps = 300000,
  [int]$MaxSteps = 200,
  [double]$TauMax = 20.0,
  [double]$LamA = 0.001,

  # A2 sampler
  [double]$Margin = 0.02,
  [double]$MinTipDist = 0.07,
  [double]$PhiMin = -1.5707963267948966,  # -pi/2
  [double]$PhiMax =  1.5707963267948966,  # +pi/2

  # PI reward
  [double]$AlphaPi = 0.0005,
  [string]$PiMetric = "tau_l1",

  # Safety filter
  [string]$SafetyFilter = "proj_box_jointlimit",
  [double]$DtauMax = 2.0,
  [double]$QMargin = 0.15,

  # Residual nominal PD
  [double]$Kp = 10.0,
  [double]$Kd = 1.0,
  [ValidateSet("auto","up","down")]
  [string]$Elbow = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "=== RUN TRAIN GRID ==="
Write-Host "Seeds: 0..4"
Write-Host "Steps: $Steps | MaxSteps(ep): $MaxSteps"
Write-Host "A2: margin=$Margin minTipDist=$MinTipDist phi=[$PhiMin,$PhiMax]"
Write-Host "TauMax=$TauMax lam_a=$LamA"
Write-Host "PI: metric=$PiMetric alpha=$AlphaPi"
Write-Host "Safety: $SafetyFilter dtau=$DtauMax q_margin=$QMargin"
Write-Host "Residual: kp=$Kp kd=$Kd elbow=$Elbow"
Write-Host "-----------------------"

# define the 4 "main" modes as argument arrays
$modes = @(
  @{ Name="baseline"; Args=@("--control","direct","--safety-filter","none") },
  @{ Name="pi_reward"; Args=@("--control","direct","--use-pi-reward","--pi-metric",$PiMetric,"--alpha-pi",$AlphaPi) },
  @{ Name="safety_filter"; Args=@("--control","direct","--safety-filter",$SafetyFilter,"--dtau-max",$DtauMax,"--q-margin",$QMargin) },
  @{ Name="residual"; Args=@("--control","residual","--safety-filter","none","--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) }
)

$seeds = 0..4

foreach ($m in $modes) {
  Write-Host ""
  Write-Host "=== Mode: $($m.Name) ==="

  foreach ($s in $seeds) {
    Write-Host ""
    Write-Host "-> Training seed=$s"
    $cmd = @(
      "python","train_rl.py",
      "--steps",$Steps,
      "--seed",$s,
      "--max-steps",$MaxSteps,
      "--tau-max",$TauMax,
      "--lam-a",$LamA,
      "--margin",$Margin,
      "--min-tip-dist",$MinTipDist,
      "--phi-min",$PhiMin,
      "--phi-max",$PhiMax
    ) + $m.Args

    Write-Host ("CMD: " + ($cmd -join " "))
    & $cmd[0] $cmd[1..($cmd.Length-1)]
  }
}

Write-Host ""
Write-Host "[OK] Treinos concluídos."
