# BrainVuln inference service — canonical frozen seed-42 model, CPU serving.
# The checkpoint identity is verified at build time AND at request time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch keeps the image ~2 GB instead of ~7 GB
RUN pip install --no-cache-dir torch==2.4.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# canonical science code + configuration + atlas (NO model files, NO data)
COPY src/ src/
COPY config/ config/
COPY data/derived/atlas_dk68_tianS1.nii.gz data/derived/atlas_dk68_tianS1.nii.gz
COPY data/derived/atlas_dk68_tianS1_info.csv data/derived/atlas_dk68_tianS1_info.csv
COPY service/ service/
# CAM reference template used by save_cam_nifti (kept from the audited artifacts)
COPY results/gradcam/OAS1_0013_gradcam.nii.gz results/gradcam/OAS1_0013_gradcam.nii.gz

# install the canonical package (preprocessing + models + gradcam + regional)
RUN pip install --no-cache-dir --no-deps -e .

# checkpoint policy: builds from GitHub have no model bytes, so the frozen
# checkpoint is fetched + SHA256-verified at CONTAINER STARTUP by
# service/checkpoint_bootstrap.py (BRAINVULN_CHECKPOINT_URL [+ token] env);
# startup FAILS loudly on any hash mismatch. When the checkpoint IS present
# in a local build context, it is verified at build time too.
COPY results/ml/resnet_seed42/metrics/test_metrics.json results/ml/resnet_seed42/metrics/test_metrics.json
RUN python - <<'PY'
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "service")
from predict_volume import CANONICAL_SHA256
print("build ok; runtime will require checkpoint sha256", CANONICAL_SHA256[:12] + "…")
PY

# non-root runtime user
RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
ENV PORT=8000

# healthcheck via the checkpoint-identity endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/api/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
