@echo off
REM =====================================================================
REM FINALIZER - run ONLY after the multi-seed campaign completes
REM (marker: results\multiseed.marker written by scripts\multiseed.bat).
REM
REM Chain: status refresh -> multi-seed analysis -> robustness report ->
REM delivery docs -> final audit -> full pytest -> checkpoint SHA check.
REM Every step writes a log; failures stop the chain loudly.
REM =====================================================================
cd /d "%~dp0.."
setlocal
set PYTHONIOENCODING=utf-8

if not exist results\multiseed.marker (
  echo Campaign not complete: results\multiseed.marker missing.
  echo Per-seed state: audit_artifacts\multiseed_status.md
  exit /b 1
)

echo [1/6] status refresh
.venv\Scripts\python.exe scripts\multiseed_status.py > audit_artifacts\multiseed_status_refresh.log 2>&1
if errorlevel 1 goto :fail

echo [2/6] multi-seed analysis (metrics, stability, CAM similarity, repeat)
.venv\Scripts\python.exe scripts\multiseed_analysis.py > results\multiseed_analysis_console.log 2>&1
if errorlevel 1 goto :fail

echo [3/6] robustness scorecard
.venv\Scripts\python.exe scripts\robustness_report.py > results\robustness_report_console.log 2>&1
if errorlevel 1 goto :fail

echo [4/6] delivery docs refresh (dashboard + FINAL_RESULTS)
.venv\Scripts\python.exe scripts\make_delivery_docs.py >> results\multiseed_analysis_console.log 2>&1
if errorlevel 1 goto :fail

echo [5/6] final project audit refresh
.venv\Scripts\python.exe scripts\make_final_audit.py >> results\multiseed_analysis_console.log 2>&1
if errorlevel 1 goto :fail

echo [6/6] full pytest + canonical checkpoint SHA
.venv\Scripts\python.exe -m pytest -q > results\final_pytest.log 2>&1
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -c "import hashlib; print('CANONICAL SHA256:', hashlib.sha256(open('results/ml/resnet_seed42/checkpoints/best.pt','rb').read()).hexdigest())"

echo FINALIZER_DONE
exit /b 0

:fail
echo FINALIZER_FAILED - check results\multiseed_analysis_console.log and results\final_pytest.log
exit /b 2
