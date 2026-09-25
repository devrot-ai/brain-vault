"""MRI deep-learning stack for BrainVuln (OASIS-1, 3D ResNet).

Research prototype — this module classifies research cohorts; it is not a
clinical diagnostic system and its outputs must not be interpreted as one.
"""

from brainvuln.mri.preprocess import preprocess_session, build_scan_index
from brainvuln.mri.dataset import OASISDataset, load_split_tables
from brainvuln.mri.models import SimpleCNN3D, ResNet18Binary, AgeSexBaseline
from brainvuln.mri.augment import training_transforms

__all__ = [
    "preprocess_session", "build_scan_index",
    "OASISDataset", "load_split_tables",
    "SimpleCNN3D", "ResNet18Binary", "AgeSexBaseline",
    "training_transforms",
]
