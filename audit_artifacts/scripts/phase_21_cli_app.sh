#!/usr/bin/env bash
# Phase 21: CLI --help for every entry point + smallest valid execution.
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8

echo "=== train_oasis1.py --help ==="
$PY scripts/train_oasis1.py --help 2>&1 | head -8; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== bridge_cnn_gene.py --help ==="
$PY scripts/bridge_cnn_gene.py --help 2>&1 | head -8; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== predict.py --help ==="
$PY predict.py --help 2>&1 | head -8; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== import_disease_map.py --help ==="
$PY scripts/import_disease_map.py --help 2>&1 | head -6; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== verify_diseases.py --help ==="
$PY scripts/verify_diseases.py --help 2>&1 | head -6; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== make_figures.py --help ==="
$PY scripts/make_figures.py --help 2>&1 | head -6; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== smallest valid inference (real checkpoint, real volume) ==="
$PY predict.py --image audit_artifacts/phase_05_example_preprocessed.nii.gz \
  --checkpoint results/ml/resnet_seed42/checkpoints/best.pt \
  --out-dir audit_artifacts/phase_21_out 2>&1 | head -10; echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== web app: import + handler smoke (no server) ==="
$PY - <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("app", "app.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    names = [n for n in dir(mod) if not n.startswith("_")]
    print("app.py imports OK; public names:", [n for n in names if n in
          ("predict_image", "demo", "run_prediction", "gradio_ui", "build_ui")][:6])
except Exception as e:
    print("app.py import FAILED:", type(e).__name__, e)
    sys.exit(1)
EOF
echo "EXIT:$?"
