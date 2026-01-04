pi_metric="power"

α=2e-5

# train (k=10)

.\run_train_baseline_pi.ps1 -Seeds (0..9) -PiMetric power -AlphaPi 0.00002 -ExpPrefix A0p

# eval

.\run_eval_baseline_pi.ps1 -TrainSeeds (0..9) -ExpPrefix A0p -PiMetric power -AlphaPi 0.00002
