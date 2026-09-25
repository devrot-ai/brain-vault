@echo off
cd /d %~dp0..
.venv\Scripts\python.exe -u scripts/bridge_cnn_gene.py 1> results\bridge_v2.log 2> results\bridge_v2.err.log
echo done > results\bridge_v2.marker
