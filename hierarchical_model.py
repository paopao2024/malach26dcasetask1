"""
HIERARCHICAL model - optimize for the hF metric directly, not flat 23-way CE.

The scoring metric is hierarchical-F with lambda=0.75: a wrong sub-class still
earns 0.75 credit IF the TOP-level class is right. Every prior approach trained
flat 23-way cross-entropy and only HOPED hF came out well. This trains FOR hF.

Three mechanisms, compounding:
  1. TWO HEADS: a 5-way top-level head + a 23-way sub head, trained jointly.
  2. HIERARCHY-WEIGHTED sub-loss: cross-top-level confusions are penalized far
     more than within-top-level ones (matching lambda=0.75 structure), so the
     model learns to at least keep the top-level right.
  3. HIERARCHICAL DECODE: combine top-head and sub-head probabilities so the
     final prediction backs off to the confident top-level when the sub-class
     is uncertain -> converts metric-zeros into metric-0.75s.

Built on the only thing that has worked: the frozen-CLAP 5-fold ensemble.
Locked-test eval, directly comparable to the 0.8226 Stage-1 ensemble.
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse

from eval_foundation import (
    CLASSES, CLS2IDX, IDX2CLS, BST, TOP_LEVELS, TOP2IDX, TOP_TO_CHILDREN_IDX,
    load_metadata, get_splits, evaluate_probs, hierarchical_f, SEED,
)

# sub-class index -> top-level index (length 23)
SUB2TOP = torch.tensor([TOP2IDX[BST[c]] for c in CLASSES], dtype=torch.long)


class SoundDataset(Dataset):
    def __init__(self, df, dataset_path):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(dataset_path) / "features" / "clap_audio_embeddings"
        self.text_dir  = Path(dataset_path) / "features" / "clap_text_embeddings"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a = np.load(self.audio_dir / f"{row.sound_id}.npy").astype(np.float32)
        t = np.load(self.text_dir  / f"{row.sound_id}.npy").astype(np.float32)
        emb = np.concatenate([a, t])
        sub = CLS2IDX[row["class"]]
        top = TOP2IDX[BST[row["class"]]]
        return torch.tensor(emb), torch.tensor(sub, dtype=torch.long), torch.tensor(top, dtype=torch.long)


class HierLoss(nn.Module):
    """Joint loss:
       - focal CE on the 5-way TOP head (getting top right = 0.75 credit floor)
       - focal CE on the 23-way SUB head, but each error WEIGHTED: a sub-error
         whose predicted-top != true-top is penalized more (weight 1.0) than a
         same-top sub-error (weight = 1 - lambda = 0.25), mirroring the metric.
    """
    def __init__(self, gamma=2.0, ls=0.1, lam=0.75, top_weight=0.5, sub2top=None):
        super().__init__()
        self.gamma = gamma; self.ls = ls; self.lam = lam
        self.top_weight = top_weight
        self.register_buffer("sub2top", sub2top)

    def _focal(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce

    def forward(self, sub_logits, top_logits, sub_t, top_t):
        # top head: standard focal
        top_loss = self._focal(top_logits, top_t).mean()

        # sub head: focal, but reweight per-sample by hierarchy severity.
        sub_focal = self._focal(sub_logits, sub_t)        # [B]
        with torch.no_grad():
            pred_sub = sub_logits.argmax(dim=1)
            pred_top = self.sub2top[pred_sub]
            same_top = (pred_top == top_t)
            # same-top mistakes are "cheap" (metric still gives 0.75); cross-top
            # mistakes are catastrophic -> weight them up.
            w = torch.where(same_top, torch.full_like(sub_focal, 1 - self.lam),
                            torch.ones_like(sub_focal))
            # correct predictions keep full weight so we still learn them
            w = torch.where(pred_sub == sub_t, torch.ones_like(w), w)
        sub_loss = (sub_focal * w).mean()

        return sub_loss + self.top_weight * top_loss


class HierMLP(nn.Module):
    def __init__(self, n_top=5, n_sub=23):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
        )
        self.sub_head = nn.Linear(128, n_sub)
        self.top_head = nn.Linear(128, n_top)

    def forward(self, x):
        h = self.trunk(x)
        return self.sub_head(h), self.top_head(h)


class HierClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, gamma=2.0, label_smoothing=0.1, lam=0.75,
                 top_weight=0.5, decode_alpha=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = HierMLP()
        self.lr = lr
        self.decode_alpha = decode_alpha   # how much the top head reshapes sub probs
        self.loss_fn = HierLoss(gamma=gamma, ls=label_smoothing, lam=lam,
                                top_weight=top_weight, sub2top=SUB2TOP.clone())
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def hier_decode(self, sub_logits, top_logits):
        """Combine heads: reweight each sub-class prob by its parent's top-prob.
        decode_alpha blends raw sub probs with top-conditioned ones."""
        sub_p = F.softmax(sub_logits, dim=1)
        top_p = F.softmax(top_logits, dim=1)
        parent_p = top_p[:, SUB2TOP.to(top_p.device)]      # [B,23] parent prob per sub
        combined = sub_p * (parent_p ** self.decode_alpha)
        combined = combined / combined.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return combined

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]

    def training_step(self, batch, _):
        x, sub, top = batch
        sl, tl = self(x)
        loss = self.loss_fn(sl, tl, sub, top)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, sub, top = batch
        sl, tl = self(x)
        self.val_outputs.append({"sub": sl.cpu(), "top": tl.cpu(), "y": sub.cpu()})

    def on_validation_epoch_end(self):
        sl = torch.cat([o["sub"] for o in self.val_outputs])
        tl = torch.cat([o["top"] for o in self.val_outputs])
        y  = torch.cat([o["y"]   for o in self.val_outputs])
        probs = self.hier_decode(sl, tl)
        _, _, hf = evaluate_probs(probs, y.tolist(), use_hierarchy=True)
        self.log_dict({"val/hF_hier": hf})
        self.val_outputs.clear()


@torch.no_grad()
def decode_probs(model, df, dataset_path, device, batch_size=256):
    dl = DataLoader(SoundDataset(df, dataset_path), batch_size=batch_size,
                    shuffle=False, num_workers=4)
    out = []
    for x, _, _ in dl:
        sl, tl = model(x.to(device))
        out.append(model.hier_decode(sl, tl).cpu())
    return torch.cat(out)


def run(dataset_path, n_epochs=50, patience=10, gamma=2.0, label_smoothing=0.1,
        top_weight=0.5, decode_alpha=0.5, ckpt_dir="checkpoints_hier"):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=SEED)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_labels = [CLS2IDX[c] for c in test_df["class"]]

    test_probs, cv = [], []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n── hier fold {fold}/5 ──")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)
        train_dl = DataLoader(SoundDataset(train_df, dataset_path), batch_size=256,
                              shuffle=True, num_workers=4)
        val_dl   = DataLoader(SoundDataset(val_df, dataset_path), batch_size=256,
                              shuffle=False, num_workers=4)
        logger = WandbLogger(project="dcase2026-task1", name=f"hier_fold{fold}")
        cbs = [
            EarlyStopping(monitor="val/hF_hier", patience=patience, mode="max"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"hier_fold{fold}",
                            monitor="val/hF_hier", mode="max", save_top_k=1,
                            save_weights_only=True, enable_version_counter=False),
        ]
        trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="auto",
                             devices=1, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=cbs, log_every_n_steps=10)
        model = HierClassifier(gamma=gamma, label_smoothing=label_smoothing,
                               top_weight=top_weight, decode_alpha=decode_alpha)
        trainer.fit(model, train_dl, val_dl)
        best = HierClassifier.load_from_checkpoint(cbs[1].best_model_path,
                                                   map_location=device).eval().to(device)
        tp = decode_probs(best, test_df, dataset_path, device)
        test_probs.append(tp)
        _, _, hf = evaluate_probs(tp, test_labels, use_hierarchy=True)
        cv.append(hf)
        print(f"  fold {fold} on locked test: {hf:.4f}")
        logger.experiment.finish()

    ens = torch.stack(test_probs).mean(dim=0)
    _, _, ens_hf = evaluate_probs(ens, test_labels, use_hierarchy=True)
    print(f"\n{'='*60}")
    print(f"HIERARCHICAL two-head model (optimizes hF directly) — locked test")
    print(f"{'='*60}")
    print(f"  top_weight={top_weight}, decode_alpha={decode_alpha}")
    print(f"  mean of 5 single models:    {float(np.mean(cv)):.4f}")
    print(f"  5-model ensemble hF_hier:   {ens_hf:.4f}")
    print(f"{'-'*60}")
    print(f"  Stage-1 ensemble reference: 0.8226")
    print(f"  gain: {ens_hf - 0.8226:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--top_weight", type=float, default=0.5)
    parser.add_argument("--decode_alpha", type=float, default=0.5)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_hier")
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs, patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing,
        top_weight=args.top_weight, decode_alpha=args.decode_alpha, ckpt_dir=args.ckpt_dir)
