@echo off
REM Detached driver: REAL MODEL EVALUATION phases 1-14
REM (validation-only threshold, frozen test evaluation, confounds, attribution)
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\eval_subject_level.py --n-boot 2000 --seed 42 > audit_artifacts\eval_subject_level_console.txt 2>&1
