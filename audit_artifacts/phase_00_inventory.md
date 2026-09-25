# Phase 0 — Repository Inventory

Audit date: 2026-09-23. Statuses here are *pending audit*; verified verdicts are
assigned per-phase in this audit and summarized in `AUDIT_REPORT.md`.

## Top-level layout

| Path | Role |
|---|---|
| `src/brainvuln/` | Library package (16 modules incl. `mri/` subpackage) |
| `scripts/` | 23 executable entry points |
| `tests/` | 11 pytest files (98 tests as of last run) |
| `configs/`, `config/` | Training + genomics + spatial configs; disease/EFO definitions |
| `data/raw/` | OASIS-1 (12 discs extracted), AHBA cache, Tian atlas |
| `data/derived/` | 128³ preprocessed volumes, atlas NIfTIs + info CSVs, AHBA expression parquet |
| `data/splits/` | matched cohort + subject-level train/val/test tables |
| `data/external/disease_maps/` | imported independent vulnerability maps + provenance |
| `results/ml/` | model runs (production `resnet_seed42` + archived v1/probes/smoke) |
| `results/bridge/` | full-scale CNN↔gene↔disease bridge outputs |
| `results/analysis/`, `results/figures/` | legacy transcriptomics results + manuscript figures |
| `app.py`, `predict.py` | Gradio web demo + inference CLI |

## Entry points (key)

* `scripts/download_oasis1.py` — OASIS-1 disc fetch/extract
* `scripts/build_oasis1_cohort.py` — metadata parse, age/sex matching, splits
* `scripts/train_oasis1.py` — 5-stage production training pipeline
* `predict.py` — single-scan inference CLI
* `app.py` — Gradio web app
* `scripts/bridge_cnn_gene.py` — CNN↔gene↔disease bridge with nulls
* `scripts/build_expression.py` — abagen AHBA expression build
* `scripts/fetch_gene_sets.py` — GWAS Catalog / Open Targets retrieval
* `scripts/import_disease_map.py` — external disease-map importer
* `scripts/make_figures.py` — 10 manuscript figures
* `scripts/collect_spec_tree.py` — spec-layout results mirror

## PROJECT INVENTORY

```
Component                  Status (pending)
-----------------------------------------
Data ingestion             data on disk (OASIS-1 12 discs, AHBA cache) — verify below
Preprocessing              182 preprocessed 128^3 volumes + preprocess module — verify below
Subject split              split tables exist — verify below
Model                      MONAI 3D ResNet-18 + SimpleCNN3D + AgeSexBaseline — verify below
Training                   train.py loop + train_oasis1.py — verify below
Validation                 val ROC-AUC selection, Youden threshold — verify below
Testing                    frozen test evaluation — verify below
Inference                  predict.py — verify below
Grad-CAM                   gradcam.py + occlusion — verify below
Atlas mapping              regional.py + DK68+TianS1 atlas — verify below
GWAS                       gwas.py (GWAS Catalog REST v2) — verify below
AHBA                       ahba.py + abagen — verify below
Gene scoring               scoring.py — verify below
Matched null               gene_sets.py — verify below
Spatial null               spatial.py (Moran/Burt/spin) — verify below
External validation        disease_maps.py + imported La Joie / Zeighami maps — verify below
CLI                        predict.py — verify below
Web app                    app.py (Gradio) — verify below
Tests                      98 passing as of last run — verify below
```
