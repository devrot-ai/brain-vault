"""Training loop (Part H): AdamW, weighted BCE, early stop on val ROC-AUC."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from brainvuln.mri.dataset import collate_items


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 40
    patience: int = 6
    warmup_epochs: int = 1
    # Cosine horizon in epochs; 0 = decay over cfg.max_epochs (legacy).
    # Set to the realistic early-stopping window (~12) so LR actually
    # anneals instead of sitting at ~97% of peak when training stops at
    # epoch 7-10 of a 40-epoch budget.
    cosine_epochs: int = 0
    # Epoch-level crash resume: when resume=True and resume_path exists,
    # continue training from the saved full optimizer state. Any prior
    # process death then costs at most one epoch, not the whole run.
    # (Bitwise determinism of dataloader shuffling after resume is NOT
    # guaranteed; the seed still fixes augmentation transforms.)
    resume: bool = False
    resume_path: str = ""
    pos_weight_auto: bool = True
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 42
    log_every: int = 20
    history: list = field(default_factory=list)


@dataclass
class TrainResult:
    best_val_auc: float
    best_epoch: int
    epochs_run: int
    history: list


def evaluate_auc(model, loader, device) -> float:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for volumes, labels, ages, sexes, _ in loader:
            logits = model(volumes.to(device))
            ps.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            ys.extend(labels.numpy().tolist())
    if len(set(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, ps))


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    checkpoint_path: str | Path | None = None,
) -> TrainResult:
    """Train with early stopping on validation ROC-AUC only.

    The test set is never touched here (Part I guarantees this by omission:
    the loader simply is not an argument).
    """
    device = torch.device(cfg.device)
    model.to(device)
    torch.manual_seed(cfg.seed)

    # pos-weight from the training partition only
    pos_weight = None
    if cfg.pos_weight_auto:
        labels = [float(train_loader.dataset.labels[r.session_id])
                  for r in train_loader.dataset.records]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(cfg.warmup_epochs, 1)
        horizon = cfg.cosine_epochs if cfg.cosine_epochs > 0 else cfg.max_epochs
        progress = (epoch - cfg.warmup_epochs) / max(
            1, horizon - cfg.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_auc, best_epoch, best_state = -1.0, -1, None
    bad = 0
    history: list = []
    start_epoch = 0
    if cfg.resume and cfg.resume_path and Path(cfg.resume_path).exists():
        st = torch.load(cfg.resume_path, map_location=device,
                        weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        start_epoch = st["next_epoch"]
        best_auc = st["best_auc"]
        best_epoch = st["best_epoch"]
        bad = st["bad"]
        best_state = st["best_state"]
        history = st["history"]
        print(f"  resumed at epoch {start_epoch} "
              f"(best val_auc {best_auc:.4f} @ {best_epoch})", flush=True)
    amp = device.type == "cuda"
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        ep_loss, seen = 0.0, 0
        for i, (volumes, labels, ages, sexes, _meta) in enumerate(train_loader):
            volumes = volumes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(volumes)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ep_loss += float(loss) * len(labels)
            seen += len(labels)
        scheduler.step()

        val_auc = evaluate_auc(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(seen, 1),
                        "val_auc": val_auc})
        marker = ""
        if val_auc == val_auc and val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "epoch": epoch,
                            "val_auc": val_auc, "config": vars(cfg)},
                           checkpoint_path)
            marker = " *"
        else:
            bad += 1
        if cfg.resume_path:
            Path(cfg.resume_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1,
                "best_auc": best_auc, "best_epoch": best_epoch,
                "bad": bad, "best_state": best_state, "history": history,
            }, cfg.resume_path)
        print(f"  epoch {epoch:02d} loss {ep_loss / max(seen, 1):.4f} "
              f"val_auc {val_auc:.4f}{marker}", flush=True)
        if bad >= cfg.patience:
            print(f"  early stop (no val-AUC improvement in {cfg.patience})",
                  flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_val_auc=best_auc, best_epoch=best_epoch,
                       epochs_run=len(history), history=history)
