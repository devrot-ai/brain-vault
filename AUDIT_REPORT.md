# BrainVuln — Full System Audit Report (Pre-Expansion)

**Audit date:** 2026-09-24 (phases 0–23 executed 2026-09-22 → 2026-09-24)
**Scope:** complete pre-expansion audit of the imaging-transcriptomics pipeline —
ML backbone (OASIS-1 3D ResNet AD classifier + Grad-CAM → M_CNN), genomics chain
(GWAS Catalog → AHBA expression → M_GENE), bridge with matched-gene + spatial
nulls, and external disease maps (La Joie atrophy, Zeighami tau).
**Standard:** every verdict below is backed by an artifact in `audit_artifacts/`
and a rerunnable driver in `audit_artifacts/scripts/`. Nothing is marked VERIFIED
without a live check. Console chokes on unicode: run scripts with
`PYTHONIOENCODING=utf-8`.

---

## 1. Verdict legend

- **VERIFIED** — executed live against real data/code, artifact on disk, checks passed.
- **PARTIALLY VERIFIED** — core checks passed; a stated caveat remains (not hidden).
- **PLACEHOLDER** — scaffolding present, no real implementation.
- **BROKEN** — fails; must fix before expansion.
- **MEASURED** — descriptive (performance) artifact, no pass/fail applicable.

## 2. Phase table

| Phase | Subject | Verdict | Key evidence (artifact) |
|---|---|---|---|
| 00 | Inventory | VERIFIED | `phase_00_inventory.md` |
| 01 | Environment / deps | VERIFIED | `phase_01_environment.txt` — 18/18 pinned deps import OK (torch 2.4.1+cpu, MONAI 1.4.0, abagen 0.1.3, pandas 1.5.3 cap, nilearn 0.14.1, brainspace 0.2.1) |
| 02 | Test suite | VERIFIED | `phase_02_tests.txt` — 98 passed, 0 failed |
| 03 | Dataset integrity | VERIFIED | `phase_03_dataset_audit.csv` + summary — 182 volumes, 0 dups, 0 missing; subject-prefix vs `_MRn` session handling correct |
| 04 | Split leakage | VERIFIED | `phase_04_split_report.md` — train 128 / val 27 / test 27, disjoint at **subject** level, 0 leakage |
| 05 | Preprocessing | VERIFIED | `phase_05_preprocessing.json` + example NIfTI — deterministic 6-step pipeline, cache bit-identical to fresh compute. Note: the saved example NIfTI carries the 2 mm affine, so `predict.py` re-preprocessing is a no-op by design |
| 06 | Model forward | VERIFIED | `phase_06_model_forward.txt` — ResNet18Binary, 33,366,609 raw params, output shape/logit sanity checked |
| 07 | Backprop/grad flow | VERIFIED | `phase_07_backprop.txt` — gradients reach all layers; MONAI `backbone.fc` is dead (no gradient) and is excluded from counts/logic |
| 08 | Training loop | VERIFIED | `phase_08_training_log.txt` + smoke checkpoint (real 8-subject, 2-epoch run) — loss decreases, checkpointing + resume work |
| 09 | Inference robustness | PARTIALLY VERIFIED | `phase_09_inference.txt` — valid input OK; 3 invalid inputs fail gracefully; **caveat:** corrupted NIfTI surfaces a raw nibabel traceback (LOW, see phase 22 R9) |
| 10 | Grad-CAM on grid | VERIFIED | `phase_10_gradcam.nii.gz/.png` + checks — saved CAM is on the reference grid by design; structured signal, top-10% mass 0.343 |
| 11 | Region mapping (M_CNN) | VERIFIED (post-fix) | `phase_11_region_mapping.tsv` — affine verdict PASS: 87/87 parcels (35/35 L, 35/35 R, 16/16 Tian). **Critical affine bug found & fixed** (see §4) |
| 12 | GWAS retrieval | VERIFIED | `phase_12_gwas_metadata.json` + `phase_12_gwas_genes.txt` — live REST v2 trait verification (MONDO_0004975, 6716 associations, "Alzheimer disease"; **EFO_0000249 returns 0 studies, verified live**); live refetch 3560 genes, overlap 3560/3560 with stored set |
| 13 | AHBA expression | VERIFIED | `phase_13_ahba_summary.json` — 87 parcels × 15,633 genes, 6 donors, 86/87 parcels sampled (1 stays NaN, honestly carried) |
| 14 | Gene map (M_GENE) | VERIFIED | `phase_14_gene_map.tsv` — 1817 genes × 86 parcels after AHBA filters |
| 15 | Matched-gene null | VERIFIED | `phase_15_null_test.tsv` — 100 nulls: no target genes, no dups, KS balance OK, p = 0.416; **fallback pool caveat** (Bing host unreachable → 15,633 AHBA-expressed background, recorded in provenance, not silent) |
| 16 | Spatial null | VERIFIED | `phase_16_spatial_null.tsv` — 50 Moran surrogates generated |
| 17 | Bridge CNN↔GENE | VERIFIED | `phase_17_cnn_gene.tsv` — explicit ID alignment, ρ_s = −0.1368, matched-gene p = 0.416 (100-null artifact; full 5000/5000 bridge below) |
| 18 | Disease validation | VERIFIED | `phase_18_disease_validation.tsv` — all 4 comparisons computed; **all ns** (honest nulls) |
| 19 | Donor robustness | VERIFIED | `phase_19_donor_robustness.tsv` — LOO over 6 donors: median ρ = −0.107, IQR [−0.138, −0.080]; consistent sign, no single donor drives the result |
| 20 | Reproducibility | VERIFIED | `phase_20_reproducibility.md` — same-seed re-run identical; different seed differs |
| 21 | CLI + web app | PARTIALLY VERIFIED | `phase_21_cli_app.md` — all `--help` paths OK; web handler exercised. **Found & mitigated:** `_get_model()` newest-by-mtime checkpoint selection was hijacked by an interrupted partial run (quarantined to `audit_artifacts/quarantined_interrupted_lightaug_run/`); risk persists → phase 22 R1 |
| 22 | Risk audit | PASS | `phase_22_risk_audit.md` — 0 TODO/FIXME/mock/placeholder/hardcoded hits; 12 risks registered (3 HIGH, 4 MEDIUM, 5 LOW) |
| 23 | Performance | MEASURED | `phase_23_performance.txt` — see §5 |
| — | Final report | THIS FILE | `AUDIT_REPORT.md` |

## 3. Production configuration (what "the model" means)

- Checkpoint: `results/ml/resnet_seed42/checkpoints/best.pt`
  (epoch 11, val ROC-AUC 0.800; ckpt keys: `config`, `epoch`, `state_dict`, `val_auc` —
  weights under `state_dict`).
- Frozen test (n = 27, threshold 0.07 frozen on val): **ROC-AUC 0.817
  [0.618, 0.972]** (subject-level bootstrap), PR-AUC 0.814, sensitivity 0.867,
  specificity 0.417, balanced accuracy 0.642, Brier 0.192. High-sensitivity
  screen operating point, not a balanced classifier (phase 22 R4).
- Test AUC reproduced **identically (0.817)** after the CAM affine fix with the
  checkpoint untouched — the bug affected only the map chain, never inference.
- Naming hazard (documented): out dir is `results/ml/{model}_seed{seed}` with
  `_lightaug`/`_noaug` suffixes when `--aug-strength != full`; a probe run with
  `--aug-strength light` writes to `resnet_seed42_lightaug`, and because of R1
  such a directory can hijack app/CLI checkpoint selection.

## 4. Critical bug found, fixed, and re-verified

**CAM affine bug (`src/brainvuln/mri/gradcam.py`).** `save_cam_nifti` wrote 128³
CAMs with an **identity affine**; resampling the atlas onto that grid treated
voxels as millimetres, so M_CNN covered only 14/87 parcels, **all
right-hemisphere**. Fix: `preprocessed_grid_affine()` derives the true 128³ grid
affine (mirrors `preprocess_session`'s resample→crop→pad); `save_cam_nifti` now
uses it. Verification: 87/87 parcels (35/35 L, 35/35 R, 16/16 Tian).

**Consequences, honestly reported:**
- Stages 3–5 re-run (`scripts/stages345.bat`, `--skip-train`, aug full): test
  AUC unchanged at 0.817; M_CNN now 87/87 nonzero parcels.
- Grad-CAM↔occlusion agreement **dropped 0.965 → 0.294** — the old value was
  inflated by the shared misalignment. M_CNN is attribution-method-sensitive
  (phase 22 R3).
- Bridge re-run on the corrected map (full 5000/5000 nulls,
  `results/bridge/bridge_results.json`): **ρ_CNN–GENE = −0.137, matched-gene
  p = 0.303, Moran p = 0.459, Burt p = 0.323, spin p = 0.862 → no significant
  CNN–transcriptomic association; concordance robust = False.** Disease maps:
  CNN↔La Joie atrophy −0.330 (p 0.31), CNN↔tau −0.451 (p 0.17), GENE↔atrophy
  −0.323, GENE↔tau −0.110 — all ns. The earlier ρ = +0.383 "spatially
  significant" result was an artifact of the affine bug; the matched-gene null
  had already refused it before the fix. **The null is the real result.**
- Old artifacts quarantined (clearly named, not deleted):
  `results/ml/resnet_seed42_misalignedCAM_{gradcam,occlusion,regional}`,
  `results/bridge_M_CNN_misaligned`.

## 5. Performance (phase 23, CPU-only, this machine)

- torch 2.4.1+cpu, 6 threads. Checkpoint 133,540,267 bytes (127.4 MB);
  `torch.load` 69 ms; full cold predict path (load → build → forward) ≈ 0.97 s.
- `preprocess_session` median **0.22 s** per volume; cached `np.load` path 4.6 ms
  (bit-identical to fresh compute, phase 5); one volume 8.4 MB as float32 [1,128,128,128].
- Inference (batch 1): median **345 ms** (min 323, max 401).
- Memory (Psapi working set): ~345 MB at start → ~813 MB after model+inference,
  ~1.0 GB peak overall. GPU numbers are **not** verifiable here (no CUDA) — phase 22 R11.

## 6. Known caveats and risks (details in `phase_22_risk_audit.md`)

HIGH:
1. **R1 — mtime-based checkpoint selection in 3 entry points** (`app.py:26`,
   `predict.py:31`, `scripts/validate_external.py:60`). A stray `results/ml/*`
   directory (probe run, interrupted run — both already observed) hijacks the
   served model. Operationally mitigated by quarantine + naming; a config-pinned
   path or metrics-floor guard is the recommended small fix.
   **FIXED (post-audit checkpoint freeze, see §9):** single canonical
   `resolve_checkpoint()` from `inference.checkpoint.path` in
   `config/analysis.yaml`; no glob/mtime anywhere; regression-tested.
2. **R2 — the genomics bridge result is null.** ρ = −0.137, all nulls ns, on the
   corrected map, at full null strength. Any summary omitting this
   misrepresents the science.
3. **R3 — M_CNN is attribution-method-sensitive.** Grad-CAM↔occlusion agreement
   0.294 post-fix; production M_CNN uses Grad-CAM alone.

MEDIUM: small test n = 27 with wide CIs (R4); sex imbalance within class,
AD ~41% M vs CN ~30% M (R5); Bing background-pool fallback for the matched-gene
null, recorded in provenance (R6); GWAS trait is MONDO_0004975 not EFO_0000249,
show_child_traits=false (R7); `preprocessed_grid_affine()` mirrors rather than
imports the preprocess geometry — a coupling kept by discipline (R8).
LOW: raw traceback on corrupt NIfTI (R9); 1/87 unsampled parcel NaN in M_GENE (R10);
CPU-only environment (R11); dead `backbone.fc` params, excluded everywhere (R12).

## 7. External validation status

`scripts/validate_external.py` and staging helpers for OASIS-3 / ADNI exist and
were code-reviewed, but **no external cohort has been executed** — no OASIS-3/ADNI
imaging access was available during the audit. External generalization is
**PARTIALLY VERIFIED** (mechanism present, not exercised end-to-end) and is the
single most valuable expansion once data access exists.

## 8. Final gate

| Component | Status |
|---|---|
| Data + splits + preprocessing | **VERIFIED** |
| ML backbone (train → infer → explain → map) | **VERIFIED** end-to-end |
| Genomics chain (GWAS → AHBA → M_GENE) | **VERIFIED** (with recorded fallback caveat R6) |
| Bridge + nulls | **VERIFIED as executed; scientific result is null** (R2) |
| External validation (OASIS-3/ADNI) | **PARTIALLY VERIFIED** (code present, not run) |
| Serving layer (app/predict) | **VERIFIED** (R1 fixed: canonical configured checkpoint, SHA256-stamped, regression-tested) |

### FINAL GATE VERDICT: **PARTIALLY READY**

The ML backbone and the genomics chain are verified end-to-end on real data; the
one critical bug found (CAM affine) was fixed and re-verified, and the corrected
results — including the null bridge — are honestly documented. The system is
**ready for expansion** under three conditions: (1) fix R1 — **DONE** (post-audit
checkpoint freeze, §9);; (2) carry the null
bridge and the Grad-CAM-only caveat (R2/R3) into every downstream claim; (3)
treat external validation as the first expansion milestone, not a checkbox.

## 9. Post-audit follow-up: checkpoint freeze (completed)

Priority-1 stabilization executed after this report was issued. The three
mtime-based checkpoint selectors were removed and replaced by one canonical
resolver:

- `config/analysis.yaml` gains `inference.checkpoint.path:
  results/ml/resnet_seed42/checkpoints/best.pt` (the audited production
  checkpoint, untouched; experimental checkpoints were not deleted).
- `src/brainvuln/config.py` gains `InferenceConfig`, the single
  `resolve_checkpoint()` (config-pinned path only; no `*.pt` search, no
  mtime; raises FileNotFoundError loudly when the configured file is
  missing), `sha256_file()`, and `checkpoint_identity()` ->
  {path, filename, sha256, size_bytes}.
- `app.py`, `predict.py`, `scripts/validate_external.py` all call that one
  resolver; no duplicate implementation exists.
- Checkpoint identity is stamped into existing manifests (no new manifest
  system): `outputs/prediction.json`, external-validation
  `{cohort}_metrics.json`, and training `test_metrics.json`.
- Regression tests (`tests/test_checkpoint_resolution.py`, 5 tests) prove:
  a newer stray checkpoint does not change resolution; missing configured
  checkpoint fails loudly; sha256 deterministic; app and CLI resolve
  identical identity; a static no-mtime/no-glob guard would fail CI if
  newest-by-mtime selection is reintroduced. Full suite: 103 passed.

Canonical checkpoint: `results/ml/resnet_seed42/checkpoints/best.pt`,
sha256 `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`,
133,540,267 bytes. Real inference re-verified end-to-end after the change
(probability finite, identity stamped, sha256 matches). Evidence:
`audit_artifacts/checkpoint_regression_test.txt`.

**One recommended next step:** execute AUDIT-REVIEW-002 — cohort design
review (counts, age/sex distributions, SMDs, leakage re-check, proposed
frozen split) and explanation-method robustness (Grad-CAM vs occlusion as
distributions over subjects), documented in
`audit_artifacts/AUDIT_REVIEW_002.md` — before any retraining decision.
