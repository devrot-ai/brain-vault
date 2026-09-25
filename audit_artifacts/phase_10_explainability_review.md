# Phase 10 (re-review): Explainability path — AUDIT-REVIEW-002 §1.2

Complete traced chain, verified live on the canonical checkpoint
(`results/ml/resnet_seed42/checkpoints/best.pt`, sha256
`5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`).

## Chain

```
MRI (OASIS-1 *_t88_masked_gfc.img/.hdr, ANALYZE pair, 176x208x176 @1mm)
  -> preprocess_session()            src/brainvuln/mri/preprocess.py
       load (nibabel, never PIL/JPEG) -> drop trailing len-1 axis
       -> nib.as_closest_canonical (RAS+ reorientation)
       -> trilinear resample to 2 mm isotropic (order=1, cval=0)
       -> tight crop to brain bbox -> centre-pad/crop to 128^3
       -> z-score inside brain (nonzero voxels), zero elsewhere
       -> float32 [1,128,128,128]
  -> ResNet18Binary.forward_features(x)   MONAI ResNet backbone
       feature tensor [1, 512, 4, 4, 4]  (16x downsampled grid)
  -> gradcam_3d()                    src/brainvuln/mri/gradcam.py
       target layer : layer4 (last conv stage of the backbone)
       target score : positive-class (AD) logit, summed over batch (=1)
       gradient source : d(score)/d(feature activations) at layer4
       channel weights : SUM over spatial dims of the gradients
                         [1,512,1,1,1]  (NOT the mean — see note)
       CAM            : ReLU( sum_c w_c * A_c )   [1,1,4,4,4]
       upsampling     : torch.nn.functional.interpolate, mode=trilinear,
                        align_corners=False, to the 128^3 input grid
       normalization  : divide by global max (if max>0) -> values in [0,1]
       thresholding   : none beyond the ReLU
       clipping       : none
  -> save_cam_nifti()
       the 128^3 CAM gets the TRUE preprocessed-grid affine
       (preprocessed_grid_affine: mirrors resample->crop->pad geometry,
       verified in the original audit), then is resampled (trilinear,
       cval=0) ONTO the T88 reference grid (176x208x176, T88/MNI affine,
       det = -1.0, RAS orientation resolved by nibabel)
  -> visualization save_planes_png()
       generated from the SAME computed CAM volume that was saved —
       axial/sagittal/coronal mid-slices, MRI in gray, CAM in jet with
       alpha=0.4 where CAM>0.05
  -> regional quantification regionalize_cam() / parcellate_nifti()
       CAM (in T88 space) trilinearly resampled onto the atlas grid,
       parcel value = mean of CAM voxels per parcel; the ATLAS is never
       interpolated; parcels outside the CAM FOV stay NaN. Full 87-parcel
       atlas used — no region selection anywhere.
```

## Notes / caveats found during this review

1. **Weight operation is a sum, not a mean** (`gradcam_3d`), while the
   alternative `GradCAMHook.cam()` uses a mean. With C=512 channels the sum
   is 512x larger, but because the final CAM is max-normalized, the two are
   identical up to a global scale factor (the normalization cancels it).
   Verified: both variants produce the same max-normalized CAM. Not a bug;
   documented for maintainers.
2. **Degenerate-input sensitivity (important).** Grad-CAM of the AD logit
   can be ALL-ZERO when the model is fed a non-canonical input. Feeding the
   raw `RAW/mpr-1_anon.hdr` scan (not the T88 masked_gfc derivative the
   pipeline is defined on) yields a confidently-CN logit (−4.77) whose
   ReLU-combined CAM is ≤0 everywhere. On the canonical T88 input the same
   subject produces a normal CAM (67% nonzero voxels). This is a property
   of Grad-CAM (ReLU can zero out a whole map), not of the checkpoint;
   `occlusion_sensitivity` does not share it. The serving entry points are
   input-agnostic, so operators must feed pipeline-conformant scans.
3. The production Grad-CAM maps on disk (27/27 subjects) are on the correct
   T88 grid (re-verified: reopening any saved NIfTI gives shape
   176x208x176 with a valid, invertible affine).

## Live verification results (this review)

- Subjects: OAS1_0013 (CN, test), OAS1_0028 (AD, test) — real T88 volumes,
  re-preprocessed bit-identically to the training cache (max abs diff 0.0).
- Run-twice determinism: probability identical to 1e-6 (exactly equal),
  CAM max abs diff exactly 0.0 for both subjects.
- Production cross-check: recomputed probabilities match the audited
  `test_predictions.csv` values (0.431390 vs 0.431389; 0.895212 exact).
- Randomization sanity: resetting all weights (seed 20240924) changes the
  CAM to NEGATIVE spatial similarity (spearman −0.238 vs trained) — the
  explanation is driven by the learned weights, not by preprocessing or a
  static structure. No critical flag.
- Grad-CAM vs occlusion (8 common subjects): per-subject Spearman
  min/median/max = 0.163/0.300/0.420; top-10 regional overlap 0–2 of 10.
  Two subjects (OAS1_0030, OAS1_0082) have degenerate all-zero occlusion
  maps (undefined correlation). Agreement is WEAK; Grad-CAM must not be
  presented as a validated biological localization.

Artifacts: `audit_artifacts/explainability/` (NIfTI + PNG + JSON),
`audit_artifacts/phase_10_region_relevance.tsv`,
`audit_artifacts/phase_10_cross_subject_stability.tsv`,
`audit_artifacts/phase_10_method_agreement_per_subject.tsv`.
