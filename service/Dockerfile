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

# require the frozen checkpoint at build time: copy it in (build context
# includes it locally) or download it (private URL). Build fails loudly on
# any mismatch. BRAINVULN_CHECKPOINT_URL can be supplied via build secret or
# build-arg; SHA256 is always enforced.
COPY results/ml/resnet_seed42/metrics/test_metrics.json results/ml/resnet_seed42/metrics/test_metrics.json
ARG BRAINVULN_CHECKPOINT_URL=""
RUN if [ -f results/ml/resnet_seed42/checkpoints/best.pt ]; then \
      echo "checkpoint present in build context"; \
    elif [ -n "$BRAINVULN_CHECKPOINT_URL" ]; then \
      BRAINVULN_CHECKPOINT_URL="$BRAINVULN_CHECKPOINT_URL" python service/checkpoint_bootstrap.py; \
    else \
      echo "FATAL: no canonical checkpoint in build context and no BRAINVULN_CHECKPOINT_URL" >&2; \
      exit 1; \
    fi && \
    python - <<'PY'
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "service")
from brainvuln.config import resolve_checkpoint
from predict_volume import CANONICAL_SHA256, _identity_checked
path = resolve_checkpoint(None)
sha = _identity_checked(path)["sha256"]
assert sha == CANONICAL_SHA256, f"checkpoint identity mismatch: {sha}"
print("build-time checkpoint verified:", path, sha[:12] + "…")
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
