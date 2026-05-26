"""
Parallel work (while BSD35k downloads) - BETTER FUSION.

Current model concatenates CLAP audio(512)+text(512) and lets a deep MLP sort
it out. Here we replace the plain concat with a GATED FUSION head that learns,
per sample, how much to rely on audio vs text before classifying. This targets
the audio-text interaction directly (text carries most of this task, but audio
matters for some classes), using the SAME precomputed embeddings -- no
re-encoding, no new data.

Measured the same way as Stage 1: 5 fold-models trained on the dev set (80%),
ensembled on the LOCKED test set (20%). Directly comparable to the Stage-1
concat ensemble (0.8226 hF_hier).

Compares two heads in one run:
  - "concat" : the Stage-1 baseline head (sanity reproduction ~0.8226)
  - "gated"  : the new gated-fusion head
so you see the delta cleanly.
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
    CLASSES, CLS2IDX, IDX2CLS,
    load_metadata, get_splits, evaluate_probs, SEED,
)


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class SoundDataset(Dataset):
    """Returns SEPARATE audio and text embeddings (not concatenated) so the
    fusion head can combine them however it likes."""
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
        return (torch.tensor(a), torch.tensor(t),
                torch.tensor(CLS2IDX[row["class"]], dtype=torch.long))


class ConcatHead(nn.Module):
    """Stage-1 baseline: concat then deep MLP."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 23),
        )

    def forward(self, a, t):
        return self.net(torch.cat([a, t], dim=1))


class GatedHead(nn.Module):
    """Gated fusion: project audio & text to a common space, compute a per-sample
    gate (how much audio vs text), blend, then classify. Also keeps the raw
    concat as a residual so it can never do worse than seeing both."""
    def __init__(self, dim=512, hidden=512, p=0.4):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p))
        self.text_proj  = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p))
        # gate looks at both modalities and outputs a weight in [0,1] per dim
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.Sigmoid(),
        )
        # classifier sees the fused vector + a compressed concat residual
        self.concat_compress = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Dropout(p))
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 23),
        )

    def forward(self, a, t):
        ap = self.audio_proj(a)
        tp = self.text_proj(t)
        g = self.gate(torch.cat([ap, tp], dim=1))   # [B, hidden] in (0,1)
        fused = g * ap + (1 - g) * tp                 # per-dim gated blend
        residual = self.concat_compress(torch.cat([a, t], dim=1))
        return self.classifier(torch.cat([fused, residual], dim=1))


class Classifier(pl.LightningModule):
    def __init__(self, head_type="gated", lr=1e-3, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = GatedHead() if head_type == "gated" else ConcatHead()
        self.lr = lr
        self.loss_fn = FocalLossWithSmoothing(gamma=gamma, label_smoothing=label_smoothing)
        self.val_outputs = []

    def forward(self, a, t):
        return self.model(a, t)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]

    def training_step(self, batch, _):
        a, t, y = batch
        loss = self.loss_fn(self(a, t), y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        a, t, y = batch
        logits = self(a, t)
        self.val_outputs.append({"logits": logits.cpu(), "labels": y.cpu(),
                                 "loss": self.loss_fn(logits, y).detach().cpu()})

    def on_validation_epoch_end(self):
        probs = F.softmax(torch.cat([o["logits"] for o in self.val_outputs]), dim=1)
        labels = torch.cat([o["labels"] for o in self.val_outputs])
        loss = torch.stack([o["loss"] for o in self.val_outputs]).mean().item()
        _, _, hf = evaluate_probs(probs, labels.tolist(), use_hierarchy=True)
        self.log_dict({"val/loss": loss, "val/hF_hier": hf})
        self.val_outputs.clear()


@torch.no_grad()
def probs_on(model, df, dataset_path, device, batch_size=256):
    dl = DataLoader(SoundDataset(df, dataset_path), batch_size=batch_size,
                    shuffle=False, num_workers=4)
    out = [F.softmax(model(a.to(device), t.to(device)), dim=1).cpu() for a, t, _ in dl]
    return torch.cat(out)


def train_ensemble(head_type, df, folds, test_df, test_labels, dataset_path,
                   device, n_epochs, patience, gamma, ls, ckpt_dir):
    test_probs, cv = [], []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)
        train_dl = DataLoader(SoundDataset(train_df, dataset_path), batch_size=256,
                              shuffle=True, num_workers=4)
        val_dl   = DataLoader(SoundDataset(val_df, dataset_path), batch_size=256,
                              shuffle=False, num_workers=4)
        logger = WandbLogger(project="dcase2026-task1", name=f"fusion_{head_type}_fold{fold}")
        cbs = [
            EarlyStopping(monitor="val/hF_hier", patience=patience, mode="max"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"{head_type}_fold{fold}",
                            monitor="val/hF_hier", mode="max", save_top_k=1,
                            save_weights_only=True, enable_version_counter=False),
        ]
        trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="auto",
                             devices=1, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=cbs, log_every_n_steps=10)
        model = Classifier(head_type=head_type, gamma=gamma, label_smoothing=ls)
        trainer.fit(model, train_dl, val_dl)
        model = Classifier.load_from_checkpoint(cbs[1].best_model_path,
                                                map_location=device).eval().to(device)
        tp = probs_on(model, test_df, dataset_path, device)
        test_probs.append(tp)
        _, _, hf = evaluate_probs(tp, test_labels, use_hierarchy=True)
        cv.append(hf)
        print(f"  [{head_type}] fold {fold} on locked test: {hf:.4f}")
        logger.experiment.finish()
    ens = torch.stack(test_probs).mean(dim=0)
    _, _, ens_hf = evaluate_probs(ens, test_labels, use_hierarchy=True)
    return float(np.mean(cv)), ens_hf


def run(dataset_path, n_epochs=50, patience=10, gamma=2.0, label_smoothing=0.1,
        ckpt_dir="checkpoints_fusion"):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=SEED)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_labels = [CLS2IDX[c] for c in test_df["class"]]

    print("=== training CONCAT baseline (sanity ~0.8226) ===")
    concat_mean, concat_ens = train_ensemble("concat", df, folds, test_df, test_labels,
                                             dataset_path, device, n_epochs, patience,
                                             gamma, label_smoothing, ckpt_dir)
    print("\n=== training GATED fusion ===")
    gated_mean, gated_ens = train_ensemble("gated", df, folds, test_df, test_labels,
                                           dataset_path, device, n_epochs, patience,
                                           gamma, label_smoothing, ckpt_dir)

    print(f"\n{'='*64}")
    print(f"FUSION COMPARISON (locked test, 5-model ensemble)")
    print(f"{'='*64}")
    print(f"{'head':<12}{'single mean':<16}{'ensemble hF_hier':<18}")
    print(f"{'-'*64}")
    print(f"{'concat':<12}{concat_mean:<16.4f}{concat_ens:<18.4f}")
    print(f"{'gated':<12}{gated_mean:<16.4f}{gated_ens:<18.4f}")
    print(f"{'-'*64}")
    print(f"gated vs concat (ensemble): {gated_ens - concat_ens:+.4f}")
    print(f"Stage-1 reference: 0.8226")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_fusion")
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs, patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing, ckpt_dir=args.ckpt_dir)
