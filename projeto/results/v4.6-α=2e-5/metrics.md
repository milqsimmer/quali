pi_metric="power"

α=2e-5

# train (k=10)

.\run_train_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00005

# eval

.\run_eval_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00005
