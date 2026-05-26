"""
Stage 1 (HONEST) - 5-model ensemble on frozen CLAP features.

Uses eval_foundation: trains 5 models on the 5 CV folds of the DEVELOPMENT set
(80%), then evaluates on the LOCKED TEST set (20%) that none of them ever saw.
Because no model trained on the test clips, the 5-model soft-vote is a TRUE,
leakage-free ensemble number (unlike the earlier 0.9152, which was leaked).

Reports on the locked test set:
  - each fold model individually
  - mean of the 5 single models
  - 5-model SOFT-VOTE ensemble   <-- the number that should beat the single mean

Model/loss/training identical to v6 (focal loss, label smoothing, cosine LR,
1024->...->23 MLP on concatenated CLAP audio+text). Frozen features => fast.
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
    CLASSES, CLS2IDX, IDX2CLS, BST,
    hierarchical_f, hierarchy_aware_predict,
    load_metadata, get_splits, evaluate_probs, SEED,
)


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
    out = [F.softmax(model(x.to(device)), dim=1).cpu() for x, _ in dl]
    return torch.cat(out)


def run(dataset_path, n_epochs=50, batch_size=256, patience=10,
        gamma=2.0, label_smoothing=0.1, ckpt_dir="checkpoints_stage1", verbose=False):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(dataset_path)
    test_idx, folds = get_splits(df, seed=SEED)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_labels = [CLS2IDX[c] for c in test_df["class"]]
    print(f"dev folds: 5 | locked test: {len(test_df)} samples\n")

    test_probs_per_model = []
    cv_scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"── fold {fold}/5 ──")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)

        train_dl = DataLoader(SoundDataset(train_df, dataset_path),
                              batch_size=batch_size, shuffle=True, num_workers=4)
        val_dl   = DataLoader(SoundDataset(val_df, dataset_path),
                              batch_size=batch_size, shuffle=False, num_workers=4)

        logger = WandbLogger(project="dcase2026-task1", name=f"stage1_fold{fold}")
        cbs = [
            EarlyStopping(monitor="val/hF_hier", patience=patience, mode="max"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"stage1_fold{fold}",
                            monitor="val/hF_hier", mode="max", save_top_k=1,
                            save_weights_only=True, enable_version_counter=False),
        ]
        trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="auto",
                             devices=1, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=cbs,
                             log_every_n_steps=10)
        model = Classifier(gamma=gamma, label_smoothing=label_smoothing)
        trainer.fit(model, train_dl, val_dl)

        # load best checkpoint back for clean test scoring
        best = cbs[1].best_model_path
        model = Classifier.load_from_checkpoint(best, map_location=device).eval().to(device)

        # score this fold's model on the LOCKED TEST SET
        tp = probs_on(model, test_df, dataset_path, device)
        test_probs_per_model.append(tp)
        _, _, hf_single = evaluate_probs(tp, test_labels, use_hierarchy=True)
        cv_scores.append(hf_single)
        print(f"  fold {fold} model on locked test: hF_hier = {hf_single:.4f}")
        logger.experiment.finish()

    stacked = torch.stack(test_probs_per_model)         # [5, Ntest, 23]
    single_mean = float(np.mean(cv_scores))
    ens_probs = stacked.mean(dim=0)                     # soft-vote
    _, _, ens_arg  = evaluate_probs(ens_probs, test_labels, use_hierarchy=False)
    _, _, ens_hier = evaluate_probs(ens_probs, test_labels, use_hierarchy=True)

    print(f"\n{'='*60}")
    print(f"STAGE 1 — HONEST ENSEMBLE (all numbers on LOCKED TEST set)")
    print(f"{'='*60}")
    for i, s in enumerate(cv_scores, 1):
        print(f"  fold {i} single model:        hF_hier = {s:.4f}")
    print(f"{'-'*60}")
    print(f"  mean of 5 single models:     hF_hier = {single_mean:.4f}")
    print(f"  5-model SOFT-VOTE ensemble:  hF_hier = {ens_hier:.4f}   (argmax {ens_arg:.4f})")
    print(f"{'-'*60}")
    gain = ens_hier - single_mean
    print(f"  ensemble gain vs single mean: {gain:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_stage1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs, patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing,
        ckpt_dir=args.ckpt_dir, verbose=args.verbose)
