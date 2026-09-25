"""Offline tests for the MRI stack (Parts D–L) on synthetic volumes.

Model/training tests run on torch CPU with tiny synthetic 3D volumes; if
torch is unavailable the whole module skips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from brainvuln.mri.augment import training_transforms  # noqa: E402
from brainvuln.mri.evaluate import (  # noqa: E402
    classification_metrics,
    select_threshold,
    subject_bootstrap_ci,
)
from brainvuln.mri.gradcam import (  # noqa: E402
    GradCAMHook,
    gradcam_3d,
    occlusion_sensitivity,
)
from brainvuln.mri.models import AgeSexBaseline, ResNet18Binary, SimpleCNN3D  # noqa: E402
from brainvuln.mri.preprocess import preprocess_session  # noqa: E402
from brainvuln.mri.regional import aggregate_regional, regionalize_cam  # noqa: E402
from brainvuln.mri.train import TrainConfig, train_model  # noqa: E402


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sphere_scan(tmp_path):
    """A synthetic 'scan' on the OASIS T88 grid: bright sphere in a 176³ box."""
    import nibabel as nib
    shape = (176, 208, 176)
    data = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    r2 = (xx - 88) ** 2 / 70 ** 2 + (yy - 104) ** 2 / 80 ** 2 + (zz - 88) ** 2 / 70 ** 2
    data[r2 < 1.0] = 0.8
    data[r2 < 0.25] = 1.6
    # an ANALYZE pair like OASIS T88 derivatives
    img = nib.Nifti1Image(data, np.diag([-1, 1, 1, 1]).astype(float))
    p = tmp_path / "OAS1_0001_MR1_mpr-1_anon_t88_masked_gfc.nii"
    nib.save(img, str(p))
    return p


# ---------------------------------------------------------------------
# Part D — preprocessing (deterministic)
# ---------------------------------------------------------------------

def test_preprocess_session_shape_and_determinism(sphere_scan):
    a = preprocess_session(sphere_scan)
    b = preprocess_session(sphere_scan)
    assert a.shape == (1, 128, 128, 128)
    assert np.array_equal(a, b)                      # fully deterministic
    assert float(a.max()) > 0.5                      # signal survived resample


def test_preprocess_zscores_inside_brain(sphere_scan):
    out = preprocess_session(sphere_scan)[0]
    brain = out != 0  # background stays exactly 0; z-scored brain is nonzero
    assert brain.any()
    assert abs(out[brain].mean()) < 1e-3
    assert abs(out[brain].std() - 1.0) < 1e-2


# ---------------------------------------------------------------------
# Part C — subject-level splits round-trip
# ---------------------------------------------------------------------

def test_load_split_tables_no_subject_overlap(tmp_path):
    from brainvuln.mri.dataset import load_split_tables
    # disjoint session sets: 20 train, 10 val, 10 test
    sessions = ([f"S{i:03d}_MR1" for i in range(20)]
                + [f"S{i:03d}_MR1" for i in range(20, 30)]
                + [f"S{i:03d}_MR1" for i in range(30, 40)])
    pd.DataFrame({"session_id": sessions[:20]}).to_csv(
        tmp_path / "train_subjects.csv", index=False)
    pd.DataFrame({"session_id": sessions[20:30]}).to_csv(
        tmp_path / "val_subjects.csv", index=False)
    pd.DataFrame({"session_id": sessions[30:]}).to_csv(
        tmp_path / "test_subjects.csv", index=False)
    tabs = load_split_tables(tmp_path)
    sets = [set(t.session_id) for t in tabs.values()]
    assert len(tabs) == 3
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


# ---------------------------------------------------------------------
# Part E — augmentation (training-only)
# ---------------------------------------------------------------------

def test_training_transforms_return_tensor_and_finiteness():
    t = training_transforms(seed=0)
    vol = torch.rand(1, 16, 16, 16)  # channel-first [C,D,H,W]
    out = t(vol)
    assert out.shape == vol.shape
    assert torch.isfinite(out).all()


def test_transforms_stochastic_between_calls_seeded():
    t = training_transforms(seed=1)
    vol = torch.rand(1, 16, 16, 16)
    a, b = t(vol), t(vol)
    assert not torch.allclose(a, b)   # stochastic augmentation on repeat


# ---------------------------------------------------------------------
# Part F — models
# ---------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [SimpleCNN3D, ResNet18Binary])
def test_models_forward_shape(model_cls):
    m = model_cls().eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1,)
    assert torch.isfinite(y).all()


def test_agesex_baseline_forward():
    m = AgeSexBaseline().eval()
    with torch.no_grad():
        y = m(torch.randn(1), torch.tensor([1.0]))
    assert y.shape == (1,)


def test_resnet_exposes_gradcam_entrypoints():
    m = ResNet18Binary()
    assert hasattr(m, "forward_features") and hasattr(m, "fc") and hasattr(m, "pool")


# ---------------------------------------------------------------------
# Part I/J — evaluation (threshold, metrics, subject-level bootstrap)
# ---------------------------------------------------------------------

def test_resume_continues_history_and_saves_state(tmp_path):
    """Epoch-crash resume: second segment continues from first's state."""
    from torch.utils.data import DataLoader, Dataset

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 2, 3, padding=1)
            self.fc = torch.nn.Linear(2, 1)

        def forward(self, x):
            pooled = torch.nn.functional.adaptive_avg_pool3d(
                torch.relu(self.conv(x)), 1).flatten(1)
            return self.fc(pooled).squeeze(-1)

    class _Fake(Dataset):
        def __init__(self, n):
            self.records = [type("R", (), {"session_id": f"s{i}"})()
                            for i in range(n)]
            self.labels = {f"s{i}": float(i % 2) for i in range(n)}

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            vol = torch.from_numpy(
                np.random.default_rng(i).normal(size=(1, 8, 8, 8))
            ).float()
            sid = f"s{i}"
            return vol, self.labels[sid], 70.0, 1.0, {"session_id": sid}

    def _collate(batch):
        vols = torch.stack([b[0] for b in batch])
        labs = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        ages = torch.tensor([b[2] for b in batch])
        sexes = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return vols, labs, ages, sexes, [b[4] for b in batch]

    train_loader = DataLoader(_Fake(8), batch_size=4, collate_fn=_collate)
    val_loader = DataLoader(_Fake(4), batch_size=4, collate_fn=_collate)
    last = tmp_path / "last_state.pt"

    def _cfg(**kw):
        base = dict(max_epochs=1, patience=99, lr=1e-3, batch_size=4,
                    warmup_epochs=0, device="cpu", seed=0,
                    pos_weight_auto=False, resume_path=str(last))
        base.update(kw)
        return TrainConfig(**base)

    r1 = train_model(_Tiny(), train_loader, val_loader, _cfg())
    assert r1.epochs_run == 1 and last.exists()

    r2 = train_model(_Tiny(), train_loader, val_loader,
                     _cfg(max_epochs=3, resume=True))
    # continuity: history carries the pre-crash epoch forward, then extends
    assert [e["epoch"] for e in r2.history] == [0, 1, 2]
    assert r2.history[0] == r1.history[0]
    assert r2.epochs_run == 3
    # state file tracks the new head
    import torch as _t
    assert _t.load(last, weights_only=False)["next_epoch"] == 3


def test_select_threshold_youden_recovers_separable_case():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
    thr = select_threshold(y, p, method="youden")
    assert 0.4 <= thr <= 0.6
    m = classification_metrics(y, p, thr)
    assert m["sensitivity"] > 0.9 and m["specificity"] > 0.9


def test_classification_metrics_confusion_and_balanced_accuracy():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.8, 0.7, 0.9])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert 0.0 <= m["roc_auc"] <= 1.0 and 0.0 <= m["brier"] <= 1.0


def test_subject_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    # 20 subjects × 3 scans; the signal lives at SUBJECT level
    y = np.repeat(rng.integers(0, 2, 20), 3)
    subj = np.repeat(np.arange(20), 3)
    p = np.clip(y * 0.6 + rng.uniform(0, 0.4, len(y)), 0, 1)
    ci = subject_bootstrap_ci(y, p, subj, threshold=0.5, n_boot=150, seed=0)
    roc = ci["roc_auc"]
    assert roc["lo"] <= roc["point"] <= roc["hi"]
    assert 0.0 <= roc["lo"] and roc["hi"] <= 1.0


# ---------------------------------------------------------------------
# Part K — Grad-CAM + occlusion
# ---------------------------------------------------------------------

def test_gradcam_3d_shape_and_range():
    # 96³ ≈ production scale: MONAI ResNet striding must leave layer4 with a
    # spatially extended map (at 32³ layer4 collapses to 1³ → degenerate CAM)
    m = ResNet18Binary().eval()
    torch.manual_seed(3)
    # Deterministic mixed-sign fc weights: a random init can give an
    # all-negative channel gradient, whose ReLU zeroes the whole CAM.
    with torch.no_grad():
        w = torch.zeros_like(m.fc.weight)
        w[:, : w.shape[1] // 2] = 1.0
        w[:, w.shape[1] // 2:] = -1.0
        m.fc.weight.copy_(w)
        m.fc.bias.zero_()
    x = torch.randn(1, 1, 96, 96, 96)
    cam = gradcam_3d(m, x)
    assert cam.shape == (96, 96, 96)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0 + 1e-6
    assert np.isfinite(cam).all()
    assert float(cam.std()) > 0.0  # spatially informative, not uniform


def test_gradcam_hook_variant_matches_reference_within_tolerance():
    m = ResNet18Binary().eval()
    torch.manual_seed(3)
    with torch.no_grad():
        w = torch.zeros_like(m.fc.weight)
        w[:, : w.shape[1] // 2] = 1.0
        w[:, w.shape[1] // 2:] = -1.0
        m.fc.weight.copy_(w)
        m.fc.bias.zero_()
    x = torch.randn(1, 1, 96, 96, 96)
    ref = gradcam_3d(m, x)
    hook = GradCAMHook(m, m.backbone.layer4)
    alt = hook.cam(x)
    # same CAM target layer; per-voxel correlation must be high
    r = np.corrcoef(ref.ravel(), alt.ravel())[0, 1]
    assert r > 0.9


def test_occlusion_sensitivity_shape_and_range():
    m = SimpleCNN3D().eval()   # small net keeps the occlusion sweep cheap
    x = torch.randn(1, 1, 32, 32, 32)
    occ = occlusion_sensitivity(m, x, patch=16, stride=8, batch_size=4)
    assert occ.shape == (32, 32, 32)
    assert float(occ.min()) >= 0.0
    assert np.isfinite(occ).all()


# ---------------------------------------------------------------------
# Part L — regional quantification
# ---------------------------------------------------------------------

def _mini_atlas(tmp_path, shape=(16, 16, 16), affine=None):
    import nibabel as nib
    affine = np.diag([2.0, 2.0, 2.0, 1.0]) if affine is None else affine
    atlas = np.zeros(shape, dtype=np.int16)
    atlas[8:, :, :] = 1
    atlas[:8, :, :] = 2
    atlas_img_path = tmp_path / "atlas.nii.gz"
    info_path = tmp_path / "atlas_info.csv"
    nib.save(nib.Nifti1Image(atlas, affine), str(atlas_img_path))
    pd.DataFrame({"id": [1, 2], "label": ["A", "B"]}).to_csv(info_path, index=False)
    return atlas_img_path, info_path


def test_regionalize_cam_on_synthetic_atlas(tmp_path):
    import nibabel as nib
    atlas_img_path, info_path = _mini_atlas(tmp_path)
    cam = np.zeros((16, 16, 16), dtype=np.float32)
    cam[8:, :, :] = 1.0
    cam_path = tmp_path / "cam.nii.gz"
    nib.save(nib.Nifti1Image(cam, np.diag([2.0, 2.0, 2.0, 1.0])),
             str(cam_path))
    s = regionalize_cam(cam_path, atlas_img_path, info_path)
    assert s.loc["A"] == pytest.approx(1.0, abs=1e-5)
    assert s.loc["B"] == pytest.approx(0.0, abs=1e-5)


def test_aggregate_regional_mean_sd_and_nan_handling():
    per_subject = {
        "S1": pd.Series({"A": 1.0, "B": 0.0}),
        "S2": pd.Series({"A": 3.0, "B": np.nan}),
    }
    table, m_cnn = aggregate_regional(per_subject)
    assert table.loc["A", "M_CNN"] == pytest.approx(2.0)
    assert table.loc["A", "n_subjects"] == 2
    assert table.loc["B", "M_CNN"] == pytest.approx(0.0)  # nan ignored, not filled
    assert m_cnn.loc["A"] == pytest.approx(2.0)
