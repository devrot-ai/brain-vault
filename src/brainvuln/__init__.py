"""BrainVuln: reproducible imaging-transcriptomics for selective vulnerability.

Pipeline: GWAS Catalog / Open Targets gene sets -> Allen Human Brain Atlas
regional expression (abagen) -> molecular brain maps -> matched-gene nulls,
independent disease-map correspondence, and spatially constrained null testing
(Moran / Burt / spin).
"""

from __future__ import annotations

import os

# On Windows, numpy's MKL runtime and PyTorch's bundled OpenMP can both be
# resident when this package is imported from a source checkout (src/ layout);
# their collision fails torch's c10.dll initialization. Setting this flag
# before either stack loads makes the coexistence explicit.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random

import numpy as np

__version__ = "0.1.0"

__all__ = ["__version__", "set_global_seed"]


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` and NumPy's global RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
