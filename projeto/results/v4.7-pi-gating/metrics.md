pi_metric="power"

α=2e-5

# train (k=10)


```powershell
python train_rl.py `
  --control direct `
  --use-pi-reward `
  --pi-metric power `
  --alpha-pi 0.00002 `
  --lam-a 0 `
  --pi-gating distance `
  --pi-gate-dist 0.25 `
  --pi-gate-min-dist 0.05 `
  --exp-id A0p__pi_power_a2e-5__gate_d0p25_dm0p05

```

# eval

```powershell
python eval_rl.py `
  --exp-id A0p__baseline_direct `
  --train-seed 0 `
  --eval-seed-base 1000 `
  --episodes 200 `
  --out-dir results
```

```powershell

python eval_rl.py `
  --exp-id A0p__pi_power_a2e-5__gate_d0p25_dm0p05 `
  --train-seed 0 `
  --eval-seed-base 1000 `
  --episodes 200 `
  --out-dir results

```
