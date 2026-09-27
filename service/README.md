# BrainVuln inference service

FastAPI wrapper around the **canonical, frozen** BrainVuln pipeline
(seed-42 `best.pt`, SHA256-verified on every start). Research use only —
not a clinical diagnostic system.

## Endpoints

| Method | Path              | Purpose                                        |
|--------|-------------------|------------------------------------------------|
| GET    | `/api/health`     | liveness + checkpoint identity (no weights)    |
| GET    | `/api/model-info` | frozen checkpoint / threshold / atlas details  |
| POST   | `/api/predict`    | multipart `file` = `.nii` / `.nii.gz` (≤200 MB)|

`/api/predict` runs the **unchanged** canonical chain:
`preprocess_session → ResNet18Binary(best.pt) → gradcam_3d → regionalize_cam`
and returns probability, frozen threshold, label, top atlas regions, and a
Grad-CAM overlay PNG (base64). Uploaded volumes exist only in a temp
directory deleted after every request.

## Run locally

```bash
# from the repo root (checkpoint already on disk)
.venv/Scripts/python.exe -m uvicorn app:app --port 8000 --app-dir service
```

Then point `site/config.js` → `window.BRAINVULN_API_BASE = "http://127.0.0.1:8000"`
and serve `site/` (e.g. `python -m http.server 8015 --directory site`).

## Deploy (Render / Railway / Cloud Run)

The 128 MB checkpoint is **not in git** and must never be public. On hosts
that build from the repo, give the build a private URL to the frozen file:

```
BRAINVULN_CHECKPOINT_URL=<private URL of best.pt>
BRAINVULN_CHECKPOINT_TOKEN=<optional bearer token>
BRAINVULN_CORS_ORIGINS=https://<your-vercel-app>.vercel.app
```

The build (and every startup) verifies the SHA256 against
`5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9` and
**refuses to serve** anything else. `render.yaml` is ready for Render's
blueprint deploy; the same Dockerfile works on Railway/Fly/Cloud Run.
Plan size must be ≥ 2 GB RAM (torch + checkpoint); CPU inference is ~5 s
per volume. Alternatively, run `docker build -f service/Dockerfile .`
with the checkpoint present in the build context — then no download is
needed and no model bytes ever leave your machine.

## Guarantees enforced in code

* checkpoint resolved only via `brainvuln.config.resolve_checkpoint`
  (no glob, no mtime, no newest-file selection)
* SHA256 verified at build time, at startup, and on every `/api/health`
* frozen threshold read from `metrics/test_metrics.json` — never re-tuned
* preprocessing version recorded in every response
