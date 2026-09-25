# AUDIT-REVIEW-002

Verification gate for explainability robustness, canonical checkpoint freeze,
and post-change code review. Executed 2026-09-24 on the canonical checkpoint
`results/ml/resnet_seed42/checkpoints/best.pt`
(sha256 `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`).
No retraining, no checkpoint modification, no test-set tuning, no new features.

## Overall Status

**PASS WITH WARNINGS**

## Explainability Status

### Grad-CAM

**PASS** (execution + determinism), with an input-conformance caveat.

- Live runs on real MRI (T88 masked_gfc volumes of test subjects
  OAS1_0013/CN and OAS1_0028/AD) through the canonical resolver; SHA256
  verified before any execution (hard abort on mismatch was armed, not
  triggered).
- Prediction finite, CAM finite, non-zero (67%/92% of voxels), non-constant.
- Saved NIfTI reopens with correct shape (176x208x176, T88 grid) and a valid
  invertible affine (det = -1.0).
- Visualization PNGs are regenerated from the same computed CAM that is
  saved (`save_planes_png(m1, ...)`) — not static images.
- Run-twice: probability identical, CAM max abs diff exactly 0.0
  (bit-stable on this CPU/torch build).
- Recomputed probabilities reproduce the audited production run:
  0.4313886 vs 0.4313900 (Δ=1.4e-6, cross-process float noise) and
  0.8952118 exact.
- **Caveat (documented, not fixed):** on a non-canonical input
  (`RAW/mpr-1_anon.hdr` instead of the pipeline's T88 masked_gfc
  derivative) Grad-CAM of the AD logit degenerates to an all-zero map for a
  confidently-CN prediction (logit -4.77; ReLU zeroes the entire map).
  This is a known Grad-CAM property (not a code bug); the production maps on
  disk were all generated from canonical inputs and are 87/87 well-formed.
  Operators must feed pipeline-conformant scans.

### Perturbation sanity

**PASS** (mechanism-connectivity only; no biological claim).

Zeroing a sphere at the CAM argmax and at a far-control locus changes the
prediction and/or attribution in every case:
- OAS1_0013: peak ΔP = -0.0140 (CAM spearman vs baseline 0.9997);
  far-control ΔP = +0.1714 (spearman 0.9659).
- OAS1_0028: peak ΔP = -0.0353 (0.9959); far-control ΔP = +0.0237 (0.9989).
The explanation mechanism is connected to the model computation (both
directions: perturbing input changes CAM; CAM responds where the model
looks). This does NOT prove regions are biologically correct.

### Randomization sanity

**PASS**. Resetting all model weights (seed 20240924) on the same input:
random-model probability 0.4978 (chance level, as expected) and the CAM
becomes spatially NEGATIVELY similar to the trained CAM
(spearman -0.238, pearson -0.410). The explanation is driven by the learned
weights, not by preprocessing structure or an implementation artifact.
No critical flag.

### Regional mapping

**PASS**. Full 87-parcel atlas (no region selection) for both audited
subjects → `audit_artifacts/phase_10_region_relevance.tsv` (174 rows:
subject, parcel_id, region, structure, mean_relevance, voxel_count).
Parcel IDs derive from the same atlas grid the CAM is resampled onto
(`parcellate_nifti`, atlas never interpolated). Production per-subject CSVs
regenerated and consistent.

### Cross-subject stability

**PASS** (as a stability measurement; explicitly not a validity claim).

From the 27 production per-subject regional CSVs:
- Top mean-relevance regions are consistent across subjects:
  caudalanteriorcingulate_L/R (mean 0.631/0.611, CV 0.61), TianS1_16
  (0.593), caudalmiddlefrontal_L (0.561), TianS1_15 (0.560).
- Per-parcel mean/median/SD/CV for all 87 parcels in
  `audit_artifacts/phase_10_cross_subject_stability.tsv`.
- CV ~0.6 at the top regions means moderate between-subject spread:
  consistently-used regions, not identical maps. Visual resemblance across
  subjects is NOT treated as biological validation.

### Explanation-method agreement

**PARTIAL** — computed and reported as distributions; agreement is weak.

Per-subject Grad-CAM vs occlusion over regional maps (8 common subjects,
87 regions each; `audit_artifacts/phase_10_method_agreement_per_subject.tsv`):
- Spearman min/median/max = 0.163 / 0.300 / 0.420
  (subjects OAS1_0030 and OAS1_0082 have degenerate all-zero occlusion maps
  → correlation undefined, excluded from the numeric distribution).
- Pearson min/median/max = 0.167 / 0.319 / 0.431.
- Top-10 regional overlap: 0–2 of 10 (median 1).
Occlusion coverage is 8/27 subjects (compute-bound); the remaining 19 are
PENDING rather than imputed. Conclusion: Grad-CAM and occlusion are NOT
interchangeable explanations of this model; neither is validated as the
"correct" one. Grad-CAM remains the production method with this caveat
carried forward.

## Checkpoint Freeze

Canonical path:
`results/ml/resnet_seed42/checkpoints/best.pt`

Canonical SHA256:
`5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`

Checkpoint identity regression:

**PASS** — new `tests/test_checkpoint_integrity.py` (3 tests):
configured-path SHA256 == manifest; identity (path/filename/size/sha)
== manifest; one-byte-mutated COPY changes SHA256 and is rejected by the
canonical check (production file untouched; temp copy auto-discarded).

Serving path consistency:

**PASS** — `app.py _get_model()`, `predict.py` CLI (full real run),
and `scripts/validate_external.py` (shared-resolver inspection; functional
run PENDING an external cohort) all resolve the canonical path + SHA256.
Proof: `audit_artifacts/scripts/review_002_checkpoint_freeze.py` output
("ALL SERVING PATHS CANONICAL: True").

Unexpected checkpoint discovery:

**NONE** — repo-wide sweep: zero `glob(*.pt)` discovery, zero mtime/newest
selection outside the sanctioned `bridge_cnn_gene.py` result-cache keying
(not checkpoint selection). The only production behaviour is: configured
path → existence check → load. `audit_artifacts/canonical_checkpoint.json`
records the full manifest (path, sha, size, architecture, seed, training
config from the checkpoint-embedded snapshot, versions, purpose).

## Diff Review

No git repository exists in this checkout, so there is no VCS diff; each
file was reviewed directly in its final state against its pre-change
behaviour (documented in the checkpoint-freeze change log).

| File | Purpose | Risk | Result |
| ---- | ------- | ---- | ------ |
| `src/brainvuln/config.py` | Add `InferenceConfig`, single `resolve_checkpoint()`, `sha256_file()`, `checkpoint_identity()` | LOW | Clean: type hints, pathlib, loud `FileNotFoundError`, no discovery, no reflection, no broad excepts. Behavior change: inference now fails loudly if the configured checkpoint is missing (intended). Training untouched. |
| `config/analysis.yaml` | Add `inference.checkpoint.path` pinning the production checkpoint | LOW | Additive section only; all existing sections byte-identical. |
| `app.py` | `_get_model()` uses shared resolver + identity; UI shows SHA256 | LOW | Behavior change: cannot silently serve a stray/newest checkpoint (intended). Threshold logic and handler flow unchanged. |
| `predict.py` | `_resolve_model_path()` delegates to shared resolver; `prediction.json` gains `checkpoint_identity`/`checkpoint_sha256` | LOW | JSON gains two keys — parsed and verified (PART 5); additive, no reader breakage. |
| `scripts/validate_external.py` | Drop mtime selection; use shared resolver; print SHA256 | LOW | Same checkpoint resolution semantics as CLI/app; external eval not yet executable (no staged cohort) — unchanged limitation. |
| `src/brainvuln/mri/external.py` | Stamp `checkpoint_identity` into `{cohort}_metrics.json` | LOW | Additive JSON key; written only at eval time (not exercised — no external data). |
| `scripts/train_oasis1.py` | Stamp `checkpoint_identity` into `test_metrics.json` | LOW | Additive key; local import kept inside the metrics block. Note: identity of a relative `--checkpoint` arg resolves against cwd — entry points run from repo root, as documented. |
| `tests/test_checkpoint_resolution.py` | 5 regression tests (newer-stray ignored, loud-missing, sha determinism, app==CLI identity, static no-mtime guard) | NONE | New file; guards against reintroducing mtime selection. |
| `tests/test_checkpoint_integrity.py` | 3 integrity tests (manifest sha, identity, mutated-copy rejection) | NONE | New file; never touches the production checkpoint. |
| `audit_artifacts/*` | Evidence artifacts (see file list below) | NONE | Documentation/measurements only. |

Windows/Linux compatibility: pathlib throughout the new code; no hardcoded
separators; UTF-8 handled explicitly; the only `subprocess` use is
predict/eval CLIs via `sys.executable` with argument lists (POSIX+Windows
safe). The `os.path.normpath` in the resolver is cross-platform.

Silent fallbacks: none found (grep for fallback/except-recovery patterns
around model loading: zero hits). Missing checkpoint ⇒ loud failure.

## Regression Tests

Tests run:
`pytest -q` (full suite) — **106 tests** (98 pre-existing + 5 checkpoint
resolution + 3 integrity), plus the targeted new-test runs shown above.

Passed:
**106**

Failed:
**0**

No existing test was deleted or weakened; three new tests were added in
this review.

## Real Inference Regression

Subject:
OAS1_0013 (test split, CN) — canonical input
`.../OAS1_0013_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0013_MR1_mpr_n4_anon_111_t88_masked_gfc.img`

Prediction:
"Alzheimer's disease" per the frozen 0.07 decision rule (probability
0.4314 ≥ 0.07). This CN subject is therefore one of the model's false
positives at the operating point — the pre-existing behaviour of the frozen
threshold, not a regression of this review. The audited production
evaluation records the true label CN with probability 0.4314.

Probability:
0.43138858675956726

Checkpoint:
`results/ml/resnet_seed42/checkpoints/best.pt`

SHA256:
`5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`
(stamped in `outputs/prediction.json` →
`audit_artifacts/explainability/prediction_OAS1_0013_T88_current.json`)

Previous result:
Same checkpoint, same canonical input, same pipeline: probability
0.4313899874687195 (production `test_predictions.csv`) and the earlier
checkpoint-freeze verification run (identical value to displayed precision).
An earlier audit-step run on the RAW `mpr-1_anon.hdr` input (probability
0.0084) is explicitly reclassified as an INVALID-INPUT run: the raw scan is
not the pipeline's canonical input, its preprocessing does not match the
training cache (correlation 0.002), and Grad-CAM on it is degenerate. That
run's record is retained as
`audit_artifacts/explainability/prediction_OAS1_0013_RAWinput_previous.json`
for the paper trail; it is superseded by the canonical-input runs.

Current result:
0.43138858675956726 — finite, stamped, SHA256 matches.

Consistency:
**PASS** (Δ vs production = 1.4e-6, relative 3.2e-6, cross-process float
variation; CAM bit-identical run-twice; prediction label identical).

## Critical Issues

None blocking this gate. (The Grad-CAM degenerate case in §Grad-CAM is a
documented input-conformance hazard, not a checkpoint or pipeline defect:
it requires feeding a non-canonical scan and it fails visibly — an all-zero
CAM — rather than silently.)

## Warnings

1. **No version control.** This checkout has no `.git`; all change history
   lives in audit narrative only. Initialize git and commit the frozen
   state before further work.
2. **Explanation-method agreement is weak** (median Spearman 0.30, top-10
   overlap ≤2/10) and occlusion covers 8/27 subjects. Downstream claims
   must carry "Grad-CAM-only, method-sensitive" (already risk R3).
3. **Frozen threshold produces a false positive on OAS1_0013** at the
   operating point (documented above) — a reminder that the 0.07 threshold
   trades specificity for sensitivity on this small test set.
4. **External validation remains PENDING** (no staged OASIS-3/ADNI data);
   `validate_external.py` was verified by inspection, not execution.
5. Training-stamp identity of a relative `--checkpoint` argument resolves
   against the process cwd; harmless under documented root-cwd usage.
6. Cross-subject stability (CV ~0.6) shows consistent-but-variable regional
   reliance; do not over-read top regions as disease foci.

## Scientific Status

- **Has the model actually been trained?** YES — on real OASIS-1 data
  (train n=128, 40-epoch budget with early stopping; production checkpoint
  epoch 11, val AUC 0.800). Verified by logs, checkpoint contents, and
  reproducible inference.
- **Held-out test evaluation performed?** YES — disjoint test n=27
  (15 AD / 12 CN), ROC-AUC 0.817 [bootstrap CI 0.618–0.972], frozen
  validation threshold 0.07, subject-level bootstrap. Verified from
  `test_metrics.json`/`test_predictions.csv`; probabilities reproduced
  bit-near-exactly in this review.
- **External validation performed?** NO — PENDING. No OASIS-3/ADNI data was
  ever available; the code path exists and is inspected but unexercised.
- **AHBA/GWAS integration validated?** PARTIALLY — the chain runs end-to-end
  on real data (live GWAS Catalog retrieval 3560 genes; AHBA 6 donors,
  86/87 parcels sampled) with one recorded fallback: the matched-gene null
  background pool uses AHBA-expressed genes because the Bing scaled-gene
  download host was unreachable (recorded in provenance, not silent).
- **Spatial-null inference validated?** The null machinery executed at full
  strength (5000 matched gene sets; Moran/Burt/spin); the scientific result
  is NULL: ρ(CNN, GENE) = -0.137, matched-null p = 0.3029, no robust
  spatial correspondence. Honest status: machinery VERIFIED, effect ABSENT.
- **Is Grad-CAM sufficiently validated?** NO — only as (a) a connected,
  deterministic mechanism (perturbation + randomization sanity pass) and
  (b) a stable regional readout. It is NOT validated as a correct or
  biologically meaningful localizer: method agreement with occlusion is
  weak, and no ground truth for localization exists here. All CAM-based
  claims must remain "regions the network used", never "regions of
  pathology".

## Final Gate

### READY FOR REAL MODEL EVALUATION

Rationale: the checkpoint is frozen and integrity-checked, every serving
path resolves it identically, the explanation pipeline is deterministic,
mechanism-connected and honestly characterized (weak method agreement is
documented, not hidden), the full regression suite passes (106/106), and
real inference reproduces the audited behaviour. The next phase is
subject-level train/validation/test evaluation (including a principled
re-derivation of the decision threshold and cohort re-review), not
additional UI work. This gate does NOT claim diagnostic capability, does
NOT claim the null bridge is a positive finding, and does NOT clear the
model for any clinical use.
