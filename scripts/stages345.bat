@echo off
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts/train_oasis1.py --aug-strength full --skip-train 1> results\train_stages345.log 2> results\train_stages345.err.log
echo done > results\stages345.marker
