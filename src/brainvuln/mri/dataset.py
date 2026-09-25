"""Datasets joining OASIS-1 splits, demographics, and preprocessed volumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from brainvuln.mri.preprocess import ScanRecord, build_scan_index, preprocess_and_cache


def load_split_tables(splits_dir: str | Path) -> dict[str, pd.DataFrame]:
    splits_dir = Path(splits_dir)
    out = {}
    for name in ("train", "val", "test"):
        p = splits_dir / f"{name}_subjects.csv"
        out[name] = pd.read_csv(p)
    return out


def load_cohort(splits_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(splits_dir) / "matched_cohort.csv")


def build_records(
    split_df: pd.DataFrame, extracted_root: str | Path
) -> list[ScanRecord]:
    """All scan records whose session id appears in ``split_df``."""
    wanted = set(split_df["session_id"].astype(str))
    return [r for r in build_scan_index(Path(extracted_root))
            if r.session_id in wanted]


@dataclass
class Item:
    volume: torch.Tensor      # [1,D,H,W] float32
    label: float
    session_id: str
    subject_id: str
    age: float
    sex: int                  # 0 = F, 1 = M


class OASISDataset(Dataset):
    """Preprocessed OASIS-1 volumes for one partition.

    One item = one scan (session). Subjects — not scans — define the split
    and the bootstrap; several items may share a subject.
    """

    def __init__(
        self,
        split_df: pd.DataFrame,
        extracted_root: str | Path,
        cache_dir: str | Path,
        *,
        augment=None,
        cohort: pd.DataFrame | None = None,
        preprocess: bool = True,
    ) -> None:
        self.records = build_records(split_df, extracted_root)
        missing = wanted = set(split_df["session_id"].astype(str)) - {
            r.session_id for r in self.records}
        if missing and preprocess:
            raise FileNotFoundError(
                f"{len(missing)} split sessions have no scan on disk "
                f"(e.g. {sorted(missing)[:3]})")
        self.cache_dir = Path(cache_dir)
        self.augment = augment
        demo = {}
        if cohort is not None:
            demo = {r.subject: (float(r.Age), 1 if str(r.Sex) == "M" else 0)
                    for r in cohort.itertuples()}
        self.demo = demo
        labels = dict(zip(split_df["session_id"].astype(str),
                          split_df["label"].astype(float)))
        self.labels = labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Item:
        rec = self.records[idx]
        path = preprocess_and_cache(rec, self.cache_dir)
        vol = np.load(path)
        vol_t = torch.from_numpy(vol)
        if self.augment is not None:
            vol_t = self.augment(vol_t)
        vol = vol_t
        age, sex = self.demo.get(rec.subject_id, (np.nan, 0))
        return Item(
            volume=vol.contiguous(),
            label=self.labels[rec.session_id],
            session_id=rec.session_id,
            subject_id=rec.subject_id,
            age=age,
            sex=sex,
        )


def collate_items(batch: list[Item]):
    volumes = torch.stack([b.volume for b in batch])
    labels = torch.tensor([b.label for b in batch], dtype=torch.float32)
    ages = torch.tensor([b.age for b in batch], dtype=torch.float32)
    sexes = torch.tensor([b.sex for b in batch], dtype=torch.float32)
    meta = {
        "session_id": [b.session_id for b in batch],
        "subject_id": [b.subject_id for b in batch],
    }
    return volumes, labels, ages, sexes, meta
