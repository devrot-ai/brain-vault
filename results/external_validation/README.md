# External validation — access and procedure

This directory receives external-cohort results. **No credentialed dataset has
been downloaded in this repository** — both cohorts below require registration
and a signed data-use agreement, which must be done by the researcher, not by
automation.

## OASIS-3 (generalization cohort)

* **Access:** free with registration at <https://www.oasis-brains.org/>
  (OASIS-3 is distributed through the same portal as OASIS-1; create an
  account, accept the data-use agreement, download the MR sessions).
* **Contents at time of writing:** 1378 participants, 2842 MR sessions.
* **Rule:** the OASIS-1 model is applied **frozen** — no tuning, no
  threshold re-selection, no retraining on OASIS-3.

## ADNI (stronger independent cohort)

* **Access:** credentialed — apply at <https://adni.loni.usc.edu/>
  (data-use agreement; access is typically granted to academic researchers).
* **Rule:** identical frozen protocol; report per-cohort metrics separately,
  never pooled with OASIS-1 for model decisions.

## Procedure once data are staged

1. Convert/stage sessions into the same layout the OASIS-1 loader expects:

   ```bash
   python scripts/stage_oasis3.py --help
   python scripts/stage_adni.py --help
   ```

2. Run the frozen evaluator (writes into this directory):

   ```bash
   python scripts/validate_external.py --help
   ```

3. Reported metrics mirror the internal test: ROC-AUC, PR-AUC, sensitivity,
   specificity, calibration, threshold (frozen from OASIS-1 validation),
   per-subject bootstrap CIs.
