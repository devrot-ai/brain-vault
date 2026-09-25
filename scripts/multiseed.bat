@echo off
REM =====================================================================
REM MULTI-SEED ROBUSTNESS driver: train seeds 1 to 4, then repeat seed 4.
REM Canonical recipe of the frozen seed-42 production run, applied
REM unchanged - ONLY the random seed differs:
REM   --epochs 40 --patience 8 --warmup 2 --cos-epochs 12 --aug-strength light
REM Checkpoint-embedded config of resnet_seed42 matches this recipe.
REM
REM Per seed: results/ml/resnet_seed<seed>/ gets its own checkpoint,
REM history, validation-only Youden threshold, test predictions and
REM regional maps. The seed-42 directory is NEVER touched. Seed 4 is
REM additionally re-run with the same seed into resnet_seed4_repeat as
REM the reproducibility check.
REM
REM Rules: model selection uses validation ROC-AUC only. The threshold is
REM derived from validation data only, per seed. No best-seed selection
REM by test AUC happens anywhere.
REM
REM Resumable: a seed whose metrics/test_metrics.json already exists is
REM skipped, so restarting this driver loses at most the in-flight seed.
REM =====================================================================
cd /d "%~dp0.."
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set RECIPE=--model resnet --epochs 40 --patience 8 --warmup 2 --cos-epochs 12 --aug-strength light

for %%S in (1 2 3 4) do (
  if exist "results\ml\resnet_seed%%S\metrics\test_metrics.json" (
    echo [seed %%S] already complete - skipping >> results\multiseed_console.log
  ) else (
    echo [seed %%S] TRAINING STARTED %date% %time% >> results\multiseed_console.log
    .venv\Scripts\python.exe -u scripts\train_oasis1.py --seed %%S %RECIPE% --dir-name resnet_seed%%S >> results\multiseed_console.log 2>&1
    echo [seed %%S] TRAINING ENDED %date% %time% >> results\multiseed_console.log
  )
)

if exist "results\ml\resnet_seed4_repeat\metrics\test_metrics.json" (
  echo [seed 4 repeat] already complete - skipping >> results\multiseed_console.log
) else (
  echo [seed 4 repeat] TRAINING STARTED %date% %time% >> results\multiseed_console.log
  .venv\Scripts\python.exe -u scripts\train_oasis1.py --seed 4 %RECIPE% --dir-name resnet_seed4_repeat >> results\multiseed_console.log 2>&1
  echo [seed 4 repeat] TRAINING ENDED %date% %time% >> results\multiseed_console.log
)

echo MULTISEED_DRIVER_DONE %date% %time% >> results\multiseed_console.log
echo done > results\multiseed.marker
