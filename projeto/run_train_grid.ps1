param(
  # Training
  [int]$Steps = 300000,
  [int]$MaxSteps = 200,
  [int[]]$Seeds = @(0,1,2,3,4),

  # Env / reward
  [double]$TauMax = 20.0,
  [double]$LamA = 0.001,
  [double]$SuccessTol = 0.05,

  # A2 sampler
  [double]$Margin = 0.02,
  [double]$MinTipDist = 0.07,
  [double]$PhiMin = -1.5707963267948966,  # -pi/2
  [double]$PhiMax =  1.5707963267948966,  # +pi/2

  # PI reward
  [double]$AlphaPi = 0.0005,
  [ValidateSet("tau_l1","power")]
  [string]$PiMetric = "tau_l1",

  # Safety filter
  [ValidateSet("none","proj_box","proj_box_jointlimit")]
  [string]$SafetyFilter = "proj_box_jointlimit",
  [double]$DtauMax = 2.0,
  [double]$QMargin = 0.15,

  # Residual nominal PD
  [double]$Kp = 10.0,
  [double]$Kd = 1.0,
  [ValidateSet("auto","up","down")]
  [string]$Elbow = "auto",

  # Grid selection
  [ValidateSet("main4","extended","factorial")]
  [string]$Grid = "extended",

  # Utility
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== RUN TRAIN GRID (TORQUE) ==="
Write-Host ("Seeds: " + ($Seeds -join ","))
Write-Host "Steps: $Steps | MaxSteps(ep): $MaxSteps | SuccessTol: $SuccessTol"
Write-Host "A2: margin=$Margin minTipDist=$MinTipDist phi=[$PhiMin,$PhiMax]"
Write-Host "TauMax=$TauMax lam_a=$LamA"
Write-Host "PI: metric=$PiMetric alpha=$AlphaPi"
Write-Host "Safety(default): $SafetyFilter dtau=$DtauMax q_margin=$QMargin"
Write-Host "Residual: kp=$Kp kd=$Kd elbow=$Elbow"
Write-Host "Grid: $Grid"
Write-Host "-----------------------"

# Base args (sempre presentes)
$baseArgs = @(
  "--steps", $Steps,
  "--max-steps", $MaxSteps,
  "--seed", 0,  # placeholder (substituído no loop)
  "--tau-max", $TauMax,
  "--lam-a", $LamA,
  "--success-tol", $SuccessTol,
  "--margin", $Margin,
  "--min-tip-dist", $MinTipDist,
  "--phi-min", $PhiMin,
  "--phi-max", $PhiMax,
  "--pi-metric", $PiMetric,
  "--alpha-pi", $AlphaPi,
  "--dtau-max", $DtauMax,
  "--q-margin", $QMargin
)

# Define variants
# Observação: não passamos --exp-id, então o train_rl.py cria exp_id automaticamente por eixo e
# agrupa seeds no mesmo diretório (um zip por seed).
$modes = @()

if ($Grid -eq "main4") {
  $modes = @(
    @{ Name="baseline_direct"; Args=@("--control","direct","--safety-filter","none") },
    @{ Name="pi_direct";       Args=@("--control","direct","--use-pi-reward","--safety-filter","none") },
    @{ Name="sf_direct";       Args=@("--control","direct","--safety-filter",$SafetyFilter) },
    @{ Name="residual";        Args=@("--control","residual","--safety-filter","none","--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) }
  )
}
elseif ($Grid -eq "extended") {
  # recomendado: mantém separação por eixo de controle, mas inclui combinações mais úteis
  $modes = @(
    # DIRECT
    @{ Name="baseline_direct";   Args=@("--control","direct","--safety-filter","none") },
    @{ Name="pi_direct";         Args=@("--control","direct","--use-pi-reward","--safety-filter","none") },
    @{ Name="sf_direct";         Args=@("--control","direct","--safety-filter",$SafetyFilter) },
    @{ Name="pi_sf_direct";      Args=@("--control","direct","--use-pi-reward","--safety-filter",$SafetyFilter) },

    # RESIDUAL
    @{ Name="baseline_residual"; Args=@("--control","residual","--safety-filter","none","--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) },
    @{ Name="pi_residual";       Args=@("--control","residual","--use-pi-reward","--safety-filter","none","--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) },
    @{ Name="sf_residual";       Args=@("--control","residual","--safety-filter",$SafetyFilter,"--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) },
    @{ Name="pi_sf_residual";    Args=@("--control","residual","--use-pi-reward","--safety-filter",$SafetyFilter,"--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) }
  )
}
else {
  # factorial: 2 (control) x 2 (pi) x 3 (sf)
  $controls = @("direct","residual")
  $pis = @($false, $true)
  $sfs = @("none","proj_box","proj_box_jointlimit")

  foreach ($c in $controls) {
    foreach ($pi in $pis) {
      foreach ($sf in $sfs) {
        $name = "c=$c__pi=$([int]$pi)__sf=$sf"
        $args = @("--control",$c,"--safety-filter",$sf)
        if ($pi) { $args += @("--use-pi-reward") }
        if ($c -eq "residual") { $args += @("--kp",$Kp,"--kd",$Kd,"--elbow",$Elbow) }
        $modes += @{ Name=$name; Args=$args }
      }
    }
  }
}

foreach ($m in $modes) {
  Write-Host ""
  Write-Host "=== Mode: $($m.Name) ==="

  foreach ($s in $Seeds) {
    Write-Host ""
    Write-Host "-> Training seed=$s"

    # copia base args e troca o seed
    $cmd = @("python","train_rl.py") + $baseArgs + $m.Args
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
