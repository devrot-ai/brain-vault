# BrainVuln — Pre-Registered Analysis Plan

This document fixes the statistical hierarchy **before** results are examined.
Any deviation must be documented as an amendment in this file (date + reason).

## 1. Primary scientific question

Does the spatial expression of genetically implicated disease genes in the
healthy human brain predict the regional pattern of vulnerability observed in
neurodegenerative disease?

Three separable hypotheses:

- **H1 (genetic enrichment).** The disease-associated gene set has a
  non-random spatial expression pattern.
- **H2 (disease correspondence).** That molecular pattern correlates with an
  independent disease-vulnerability map.
- **H3 (spatial robustness).** That correspondence survives spatially
  constrained null models.

## 2. Gene definitions (two tiers, one primary)

- **Tier A (primary).** GWAS Catalog REST v2, explicit EFO id,
  `show_child_traits` and `extended_geneset` recorded per retrieval
  (defaults: false/false). Full pagination; retrieval metadata saved.
- **Tier B (robustness).** Open Targets *genetic* evidence class,
  overall association score ≥ 0.10 (primary) and ≥ 0.30 (stringent).
- **Extension (offline).** Open Targets Genetics locus-to-gene / fine-mapping
  gene sets from downloadable OTG data — not available via the Platform API;
  documented as a future amendment, never fabricated.
- Sensitivity axis: `gwascat` vs `opentargets` vs `union` (`--gene-definition`).

## 3. Transcriptomic processing (primary settings)

abagen `get_expression_data`: `probe_selection=diff_stability`,
`donor_probes=aggregate`, `region_agg=donors`, `agg_metric=mean`,
`corrected_mni=True`, `reannotated=True`, `sample_norm=srs`, `gene_norm=srs`,
`missing=None`. Parcels without adequate AHBA sampling stay missing and are
dropped once, explicitly, before all analyses. No imputation in the primary
analysis; filled variants are sensitivity runs only.

Anatomical space: volumetric union of Desikan-Killiany 68 (cortex) and
Tian 2020 Subcortex S1 (MNI152), non-overlapping, with DK68 surface GIFTIs
retained for cortical spin tests. Atlas sensitivity: DK68-only cortical rerun.

## 4. Primary analysis and statistical hierarchy

```
regional gene score  S_r = (1/|G|) Σ z_{r,g}      (z across parcels)
        -> H1: matched-gene nulls (1000 sets; 1:1 greedy matching without
           replacement on mean expression + expression variance; KS balance
           diagnostics reported) -> empirical p on the peak/range statistics
        -> H2: Spearman rho(S_r, independent disease map)
        -> H3: spatial nulls — Moran spectral randomization (primary) and
           Burt-2020 variogram surrogates (sensitivity), 5000 surrogates,
           two-sided empirical p = (1 + #{|r_null| ≥ |r_obs|}) / (1 + N)
        -> BH-FDR across the disease family at q = 0.05
        -> concordance flag: result is "robust" when ≥2 null families agree
```

Cortical spin permutations (Alexander-Bloch 2018) are a cortical-surface
sensitivity analysis for real (non-synthetic) maps.

## 5. Negative controls (pre-registered; do not edit after unblinding)

1. **Matched gene sets** must not reproduce the disease map better than the
   true set (observed rho in the far tail of the matched-null distribution).
2. **Cross-disease gene sets**: an unrelated disease's gene set (e.g. PD genes
   vs the AD map) should show systematically weaker correspondence.
3. **Unrelated phenotype map** (supplied under
   `data/external/negative_controls/`): no robust correspondence.

## 6. Robustness axes (each re-run with one knob changed)

- Donors: leave-one-donor-out (6 folds); report median r and IQR.
- Atlas: DK68-only cortical; later Schaefer-based union.
- Gene definition: Tier A vs Tier B (0.10 / 0.30) vs union.
- Score method: mean_z (primary) vs mean_rank vs first_pc vs p-weighted.
- Spatial null: Moran vs Burt-2020 vs spin (where applicable).

## 7. Multi-disease stage

Progression: AD (pilot) → AD+PD+HD → ~15 disorders. The D×R matrix supports:

- shared vulnerability (parcels in the top decile across diseases),
- disease specificity (top decile unique to one disease),
- disease–disease similarity (spatially-corrected Spearman + Ward clustering
  on correlation distance — clusters are discovered, then tested),
- genetic convergence (map similarity with low gene-set Jaccard overlap).

## 8. Deliverables per run

`results/analysis/run_primary_results.json` + manifests (config hash, git
commit, package versions, seeds, retrieval dates) + figures. Every number in a
manuscript must be traceable to a manifest.
