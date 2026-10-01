#!/usr/bin/env bash
# Full pipeline re-run (after the scale-bar mask fix). Main results first, ablations after.
# Usage (from the project root): bash run_all.sh [python]
PY=${1:-.venv-gpu/Scripts/python}
step() { echo "=== $(date +%H:%M) $*"; "$PY" "$@" > /dev/null 2>&1; rc=$?; echo "    exit=$rc"; [ $rc -eq 0 ] || { echo "STOPPED at $*"; exit $rc; }; }
step 04_preprocess.py
step 05_train_autoencoder.py
step 06_evaluate_autoencoder.py
step 07_train_deep_svdd.py
step 08_evaluate_deep_svdd.py
step 09_train_vae.py
step 12_train_mkd.py
step 10_compare_models.py
step 11_visualize_results.py
step mid_review_val_summary.py
echo "=== $(date +%H:%M) MAIN RESULTS READY"
for v in expanded naive; do for m in ae svdd_frozen svdd_ft; do step 13_ablation_retrain.py --variant $v --model $m; done; done
step 10_compare_models.py
step 11_visualize_results.py
step 14_audit.py
echo "=== $(date +%H:%M) ALL DONE"
