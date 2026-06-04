"""
Stage-1 CLAP ensemble -- OFFICIAL-METRIC version.

Trains 5 fold models on frozen CLAP (audio + text) features and reports a
soft-vote ensemble on the locked 20% test set. Identical model / loss /
schedule to the original stage1_ensemble.py, with two methodologically
important changes:

  1. Validation and test scoring use the ORGANIZERS' metric (macro_hPRF from
     a_official_metric, which is byte-for-byte their evaluate.py).
  2. Predictions are plain argmax. The earlier hierarchy_aware_predict trick
     is not part of the official pipeline and has been removed.

Checkpoints are written to a NEW directory (checkpoints_stage1_official) so
the original checkpoints_stage1/ models are preserved.

Run on the JKU server (ssh), in tmux (training takes a while across 5 folds,
even with frozen features):
    tmux new -s a_stage1
    source ~/.bashrc && conda activate malach
    python a_stage1_ensemble.py --dataset_path ~/data/bsd10k
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

from eval_foundation import (
    CLASSES, CLS2IDX, IDX2CLS,
    load_metadata, get_splits, SEED,
)
from a_official_metric import macro_hPRF


# ──────────────────────────────────────────────────────────────────────
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class SoundDataset(Dataset):
    def __init__(self, df, dataset_path):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(dataset_path) / "features" / "clap_audio_embeddings"
        self.text_dir  = Path(dataset_path) / "features" / "clap_text_embeddings"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = np.load(self.audio_dir / f"{row.sound_id}.npy").astype(np.float32)
        text  = np.load(self.text_dir  / f"{row.sound_id}.npy").astype(np.float32)
        emb   = np.concatenate([audio, text])
        return torch.tensor(emb), torch.tensor(CLS2IDX[row["class"]], dtype=torch.long)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,  256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256,  128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,   23),
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP()
        self.lr = lr
        self.loss_fn = FocalLossWithSmoothing(gamma=gamma, label_smoothing=label_smoothing)
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sch]

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.val_outputs.append({"logits": logits.cpu(), "labels": y.cpu(),
                                 "loss": self.loss_fn(logits, y).detach().cpu()})

    def on_validation_epoch_end(self):
        # Plain argmax predictions, scored with the OFFICIAL metric.
        logits = torch.cat([o["logits"] for o in self.val_outputs])
        labels = torch.cat([o["labels"] for o in self.val_outputs])
        loss = torch.stack([o["loss"] for o in self.val_outputs]).mean().item()

        preds = logits.argmax(dim=1).tolist()
        true_strs = [IDX2CLS[int(l)] for l in labels.tolist()]
        pred_strs = [IDX2CLS[int(p)] for p in preds]
        hP, hR, hF = macro_hPRF(true_strs, pred_strs, lambda_param=0.75)

        self.log_dict({"val/loss": loss,
                       "val/hP": hP, "val/hR": hR, "val/hF": hF},
                      prog_bar=False)
        self.val_outputs.clear()


@torch.no_grad()
def probs_on(model, df, dataset_path, device, batch_size=256):
    dl = DataLoader(SoundDataset(df, dataset_path), batch_size=batch_size,
                    shuffle=False, num_workers=4)
    out = [F.softmax(model(x.to(device)), dim=1).cpu() for x, _ in dl]
    return torch.cat(out)


def run(dataset_path, n_epochs=50, batch_size=256, patience=10,
        gamma=2.0, label_smoothing=0.1,
        ckpt_dir="checkpoints_stage1_official", verbose=False):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=SEED)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_true_strs = [c for c in test_df["class"]]
    print(f"dev folds: 5 | locked test: {len(test_df)} samples\n")

    test_probs_per_model = []
    cv_test_scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"── fold {fold}/5 ──")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)

        train_dl = DataLoader(SoundDataset(train_df, dataset_path),
                              batch_size=batch_size, shuffle=True, num_workers=4)
        val_dl   = DataLoader(SoundDataset(val_df, dataset_path),
                              batch_size=batch_size, shuffle=False, num_workers=4)

        logger = WandbLogger(project="dcase2026-task1",
                             name=f"a_stage1_official_fold{fold}")
        cbs = [
            EarlyStopping(monitor="val/hF", patience=patience, mode="max"),
            ModelCheckpoint(dirpath=str(ckpt_dir),
                            filename=f"a_stage1_official_fold{fold}",
                            monitor="val/hF", mode="max", save_top_k=1,
                            save_weights_only=True, enable_version_counter=False),
        ]
        trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="auto",
                             devices=1, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=cbs,
                             log_every_n_steps=10)
        model = Classifier(gamma=gamma, label_smoothing=label_smoothing)
        trainer.fit(model, train_dl, val_dl)

        # Load best (highest val/hF) checkpoint and score on locked test set.
        best = cbs[1].best_model_path
        model = Classifier.load_from_checkpoint(best, map_location=device).eval().to(device)

        tp = probs_on(model, test_df, dataset_path, device)
        test_probs_per_model.append(tp)
        pred_strs = [IDX2CLS[int(i)] for i in tp.argmax(dim=1)]
        hP, hR, hF = macro_hPRF(test_true_strs, pred_strs, lambda_param=0.75)
        cv_test_scores.append(hF)
        print(f"  fold {fold} on locked test : hP={hP:.4f}  hR={hR:.4f}  hF={hF:.4f}")
        logger.experiment.finish()

    # ---- 5-model soft-vote ensemble ----
    stacked = torch.stack(test_probs_per_model)            # [5, Ntest, 23]
    ens_probs = stacked.mean(dim=0)
    ens_pred_strs = [IDX2CLS[int(i)] for i in ens_probs.argmax(dim=1)]
    ens_hP, ens_hR, ens_hF = macro_hPRF(test_true_strs, ens_pred_strs, lambda_param=0.75)
    single_mean = float(np.mean(cv_test_scores))

    print(f"\n{'='*60}")
    print(f"STAGE 1 -- OFFICIAL METRIC (all numbers on LOCKED TEST set)")
    print(f"{'='*60}")
    for i, s in enumerate(cv_test_scores, 1):
        print(f"  fold {i} single model        hF = {s:.4f}")
    print(f"{'-'*60}")
    print(f"  mean of 5 single models     hF = {single_mean:.4f}")
    print(f"  5-model SOFT-VOTE ensemble  hP = {ens_hP:.4f}")
    print(f"                              hR = {ens_hR:.4f}")
    print(f"                              hF = {ens_hF:.4f}    <- leaderboard")
    print(f"{'-'*60}")
    print(f"  ensemble gain vs single mean: {ens_hF - single_mean:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_stage1_official")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs, patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing,
        ckpt_dir=args.ckpt_dir, verbose=args.verbose)
