@echo off
rem Production retrain after the LR/augmentation diagnosis (sweep_lr.bat).
rem Recipe: LR 1e-4 (best of {1e-4,3e-4,1e-3}), 2-epoch warmup, cosine
rem annealed over 12 epochs so LR actually decays inside the realistic
rem early-stopping window, light augmentation so train loss can descend
rem below the full-aug ~0.6 plateau without the no-aug overfit collapse.
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts/train_oasis1.py --epochs 40 --patience 8 --warmup 2 --cos-epochs 12 --aug-strength light --resume-epochs > results\train_production_v2.log 2> results\train_production_v2.err.log
echo done > results\train_production_v2.marker
