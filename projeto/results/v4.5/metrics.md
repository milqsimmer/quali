pi_metric="power"

# train
.\run_train_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00005
.\run_train_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00002
.\run_train_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00001

# eval
.\run_eval_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00005
.\run_eval_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00002
.\run_eval_baseline_pi.ps1 -ExpPrefix A0p -PiMetric power -AlphaPi 0.00001
