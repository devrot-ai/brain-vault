#!/usr/bin/env bash
# Diagnostic probes for the epoch-1 val peak. Known baseline (LR 1e-4):
#   train loss plateaus ~0.59-0.67 for all 8 epochs -> from-scratch underfitting
#   best val_auc 0.8111 at epoch 1, never exceeded.
# Probes (4 epochs each, patience disabled so they always run all 4):
#   P1: lr 3e-4          - does more LR alone break the plateau?
#   P2: lr 1e-3          - more aggressive
#   P3: lr 1e-3 no-aug   - isolates augmentation's drag on fitting
cd "$(dirname "$0")/.."
./.venv/Scripts/python.exe -u scripts/train_oasis1.py --lr 3e-4 --epochs 4 \
  --patience 99 --tag lr3e4_probe \
  > results/sweep_lr3e4.log 2> results/sweep_lr3e4.err.log
./.venv/Scripts/python.exe -u scripts/train_oasis1.py --lr 1e-3 --epochs 4 \
  --patience 99 --tag lr1e3_probe \
  > results/sweep_lr1e3.log 2> results/sweep_lr1e3.err.log
./.venv/Scripts/python.exe -u scripts/train_oasis1.py --lr 1e-3 --epochs 4 \
  --patience 99 --no-augment --tag lr1e3_noaug_probe \
  > results/sweep_lr1e3_noaug.log 2> results/sweep_lr1e3_noaug.err.log
echo done > results/sweep_done.marker
