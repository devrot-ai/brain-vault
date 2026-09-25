@echo off
rem Remaining LR probes (probe 1 / lr3e-4 completed before the restart).
rem Launched via PowerShell Start-Process so it survives session restarts.
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts/train_oasis1.py --lr 1e-3 --epochs 4 --patience 99 --tag lr1e3_probe > results\sweep_lr1e3.log 2> results\sweep_lr1e3.err.log
.venv\Scripts\python.exe -u scripts/train_oasis1.py --lr 1e-3 --epochs 4 --patience 99 --no-augment --tag lr1e3_noaug_probe > results\sweep_lr1e3_noaug.log 2> results\sweep_lr1e3_noaug.err.log
echo done > results\sweep_done.marker
