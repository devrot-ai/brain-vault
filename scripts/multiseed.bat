@echo off
REM =====================================================================
REM Multi-seed robustness driver - REAL MODEL EVALUATION, phase 15.
REM PREPARED BUT NOT EXECUTED (STOP condition: only launch AFTER the
REM primary test result is recorded in
REM audit_artifacts/subject_level_evaluation.md).
REM
REM Launch with:   scripts\multiseed.bat
REM
REM Trains seeds 1-4 through the tested scripts/train_oasis1.py path
REM (seed 42 = canonical baseline; its frozen run in
REM results/ml/resnet_seed42/ is NEVER re-trained or touched), then
REM aggregates per-seed test metrics + attribution agreement into
REM results/multi_seed/.
REM
REM Rules: each seed gets its own training run, its own validation
REM threshold, its own checkpoint; NEVER pick the "best" seed using test
REM metrics - this is a robustness report, not a model-selection contest.
REM =====================================================================
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\seed_robustness.py --seeds 1 2 3 4 --epochs 30 > results\multi_seed_console.log 2>&1
echo Done. See results\multi_seed_console.log and results\multi_seed\summary.json
