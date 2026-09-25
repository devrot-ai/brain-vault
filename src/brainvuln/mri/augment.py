"""Training-only 3D augmentation (Part E) with biological justification.

Every transform below perturbs the image in a way that **preserves brain
anatomy and diagnosis-relevant morphology**. Validation/test data is NEVER
augmented — the deterministic preprocessing pipeline is the whole transform.

* **Small rotations (±7°)** — head positioning in the scanner varies between
  sessions; the same brain could legitimately have been acquired at any small
  angle. Rotation does not change cortical thickness or atrophy.
* **Small translations (±3 voxels = ±6 mm)** — the crop/pad placement is
  arbitrary within the acquisition field of view; a few voxels of shift is
  within realistic repositioning variability.
* **Small scaling (±5%)** — subjects' heads differ in size and the affine
  atlas registration leaves residual scale error; 5% does not alter
  diagnosis-relevant proportions.
* **Mild intensity scaling (±10%) and shifting (±0.1σ)** — scanner gain and
  RF inhomogeneity vary between sessions; the disease signal is morphometric,
  not absolute intensity.
* **Gaussian noise (σ=0.02 of brain intensity range)** — MR noise is
  approximately Gaussian (Rice-distributed magnitude, near-Gaussian at these
  SNRs); small noise regularizes without destroying fine structure.

**No left-right flip.** Brain asymmetry is real (planum temporale, petalia)
and Alzheimer-related atrophy is asymmetric in progression; a mirrored brain
is *not* a plausible image of the same patient, so flipping would teach the
network features that never occur. (It is a reasonable choice for symmetric
tasks; we deliberately do not use it here.)
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


def training_transforms(
    *,
    rotate_range: float = 7.0 * np.pi / 180.0,
    translate_range: float = 3.0,
    scale_range: float = 0.05,
    intensity_scale: float = 0.10,
    intensity_shift: float = 0.10,
    noise_sigma: float = 0.02,
    seed: int | None = None,
) -> Compose:
    """MONAI transforms for training tensors [1,D,H,W]; tensor in, tensor out.

    Internally uses the dictionary variants (wrapped/unwrapped here) so the
    dataset can call ``augment(vol_t)`` directly.
    """
    compose = Compose([
        RandAffined(
            keys="img",
            prob=0.9,
            rotate_range=(rotate_range,) * 3,
            translate_range=(translate_range,) * 3,
            scale_range=(scale_range,) * 3,
            mode="bilinear",
            padding_mode="zeros",
            device=torch.device("cpu"),
        ),
        RandScaleIntensityd(keys="img", factors=intensity_scale, prob=0.5),
        RandShiftIntensityd(keys="img", offsets=intensity_shift, prob=0.5),
        RandGaussianNoised(keys="img", prob=0.3, std=noise_sigma),
    ])
    compose.set_random_state(seed)

    def _apply(tensor: torch.Tensor) -> torch.Tensor:
        out = compose({"img": tensor})
        return out["img"]

    return _apply
