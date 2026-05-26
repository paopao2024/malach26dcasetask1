"""
DCASE 2026 Task 1 - v6 + checkpoint saving
Identical model/training to the original v6 (focal loss, label smoothing,
bigger MLP, cosine LR) on FROZEN CLAP audio+text features.

ONLY addition: saves each fold's best model to a clean, predictable path
  ./checkpoints/v6_fold{1..5}.ckpt
so the Stage-1 ensemble script can load exactly these five (instead of
hunting through wandb run-hash folders).

Frozen features => all 5 folds train in minutes. This re-confirms the
0.8048 baseline AND produces the artifacts every later stage builds on.
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
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
import argparse


BST = {
    "m-sp": "m",  "m-si": "m",  "m-m": "m",
    "is-p": "is", "is-s": "is", "is-w": "is", "is-k": "is", "is-e": "is",
    "sp-s": "sp", "sp-c": "sp", "sp-p": "sp",
    "fx-o": "fx", "fx-v": "fx", "fx-m": "fx", "fx-h": "fx",
    "fx-a": "fx", "fx-n": "fx", "fx-ex": "fx", "fx-el": "fx",
    "ss-n": "ss", "ss-i": "ss", "ss-u": "ss", "ss-s": "ss",
}
CLASSES = sorted(BST.keys())
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
TOP_LEVELS = sorted(set(BST.values()))
TOP2IDX = {t: i for i, t in enumerate(TOP_LEVELS)}
TOP_TO_CHILDREN_IDX = {
    top: [CLS2IDX[c] for c in CLASSES if BST[c] == top]
    for top in TOP_LEVELS
}


def hierarchical_f(true_labels, pred_labels, lam=0.75):
    total = 1.0 + lam
    hp_per_class = defaultdict(list)
    hr_per_class = defaultdict(list)
    for true, pred in zip(true_labels, pred_labels):
        true_set = {true: 1.0, BST[true]: lam}
        pred_set = {pred: 1.0, BST[pred]: lam}
        overlap = sum(min(true_set.get(k, 0), pred_set.get(k, 0))
                      for k in set(true_set) | set(pred_set))
        hp_per_class[pred].append(overlap / total)
        hr_per_class[true].append(overlap / total)
    all_hp, all_hr = [], []
    for cls in CLASSES:
        hp = np.mean(hp_per_class[cls]) if hp_per_class[cls] else 0.0
        hr = np.mean(hr_per_class[cls]) if hr_per_class[cls] else 0.0
        all_hp.append(hp)
        all_hr.append(hr)
    hp = float(np.mean(all_hp))
    hr = float(np.mean(all_hr))
    hf = 2 * hp * hr / (hp + hr) if (hp + hr) > 0 else 0.0
    return hp, hr, hf


def hierarchy_aware_predict(logits):
    probs = F.softmax(logits, dim=1)
    top_probs = torch.zeros(logits.size(0), len(TOP_LEVELS), device=logits.device)
    for top in TOP_LEVELS:
        children_idx = TOP_TO_CHILDREN_IDX[top]
        top_probs[:, TOP2IDX[top]] = probs[:, children_idx].sum(dim=1)
    adjusted = probs.clone()
    for i, cls in enumerate(CLASSES):
        parent_idx = TOP2IDX[BST[cls]]
        adjusted[:, i] = probs[:, i] * top_probs[:, parent_idx]
    return adjusted.argmax(dim=1)


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.ls)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


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
        label = CLS2IDX[row["class"]]
        return torch.tensor(emb), torch.tensor(label, dtype=torch.long)


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
    def __init__(self, lr=1e-3, gamma=2.0, label_smoothing=0.1, verbose=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP()
        self.lr = lr
        self.loss_fn = FocalLossWithSmoothing(gamma=gamma, label_smoothing=label_smoothing)
        self.verbose = verbose
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_outputs.append({
            "loss":   loss.detach().cpu(),
            "logits": logits.cpu(),
            "labels": y.cpu(),
        })

    def on_validation_epoch_end(self):
        all_loss   = torch.stack([o["loss"]   for o in self.val_outputs])
        all_logits = torch.cat ([o["logits"] for o in self.val_outputs])
        all_labels = torch.cat ([o["labels"] for o in self.val_outputs])
        true_str = [IDX2CLS[l.item()] for l in all_labels]

        preds_orig = all_logits.argmax(dim=1)
        pred_orig_str = [IDX2CLS[p.item()] for p in preds_orig]
        hp_o, hr_o, hf_o = hierarchical_f(true_str, pred_orig_str)

        preds_hier = hierarchy_aware_predict(all_logits)
        pred_hier_str = [IDX2CLS[p.item()] for p in preds_hier]
        hp_h, hr_h, hf_h = hierarchical_f(true_str, pred_hier_str)

        self.log_dict({
            "val/loss":    all_loss.mean().item(),
            "val/hF":      hf_o,
            "val/hF_hier": hf_h,
        })

        if self.verbose or (self.current_epoch + 1) % 10 == 0:
            print(f"    epoch {self.current_epoch+1:3d}  "
                  f"loss={all_loss.mean():.4f}  "
                  f"hF={hf_o:.4f}  hF_hier={hf_h:.4f}")

        self.val_outputs.clear()


def run(dataset_path, n_epochs=50, batch_size=256, seed=42,
        early_stop_patience=10, gamma=2.0, label_smoothing=0.1,
        ckpt_dir="checkpoints", run_tag="v6", verbose=False):
    pl.seed_everything(seed, workers=True)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")
    print(f"config: focal_gamma={gamma}, label_smoothing={label_smoothing}, "
          f"max_epochs={n_epochs}, patience={early_stop_patience}")
    print(f"checkpoints -> {ckpt_dir}/v6_fold{{1..5}}.ckpt\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class"]), 1):
        print(f"\n── fold {fold}/5 ──────────────────────────────")
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_dl = DataLoader(
            SoundDataset(train_df, dataset_path),
            batch_size=batch_size, shuffle=True, num_workers=4,
        )
        val_dl = DataLoader(
            SoundDataset(val_df, dataset_path),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

        logger = WandbLogger(project="dcase2026-task1", name=f"{run_tag}_fold{fold}")
        early_stop = EarlyStopping(monitor="val/hF_hier", patience=early_stop_patience,
                                   mode="max", verbose=False)
        # Save the single best (by val/hF_hier) checkpoint for this fold to a
        # clean, predictable filename the ensemble script can find.
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{run_tag}_fold{fold}",
            monitor="val/hF_hier", mode="max",
            save_top_k=1, save_weights_only=True, enable_version_counter=False,
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            logger=logger,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[early_stop, ckpt_cb],
            log_every_n_steps=10,
        )

        model = Classifier(gamma=gamma, label_smoothing=label_smoothing, verbose=verbose)
        trainer.fit(model, train_dl, val_dl)

        hf_orig = trainer.callback_metrics.get("val/hF",      torch.tensor(0)).item()
        hf_hier = trainer.callback_metrics.get("val/hF_hier", torch.tensor(0)).item()
        best_path = ckpt_cb.best_model_path
        results.append({"fold": fold, "hF": hf_orig, "hF_hier": hf_hier})
        print(f"  fold {fold} done   hF={hf_orig:.4f}   hF_hier={hf_hier:.4f}")
        print(f"  saved -> {best_path}")
        logger.experiment.finish()

    print(f"\n{'='*60}")
    print(f"SUMMARY ({run_tag})")
    print(f"{'='*60}")
    print(f"{'Fold':<6}{'hF (argmax)':<15}{'hF (hier)':<15}")
    print(f"{'-'*36}")
    for r in results:
        print(f"{r['fold']:<6}{r['hF']:<15.4f}{r['hF_hier']:<15.4f}")
    print(f"{'-'*36}")
    mean_hf      = np.mean([r['hF']      for r in results])
    mean_hf_hier = np.mean([r['hF_hier'] for r in results])
    print(f"{'MEAN':<6}{mean_hf:<15.4f}{mean_hf_hier:<15.4f}")
    print(f"{'='*60}")
    print(f"expected ~0.8048 (per-fold mean). Ensemble of these 5 should beat it.")
    print(f"next: python ensemble_v6.py --dataset_path {dataset_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--tag", type=str, default="v6")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run(args.dataset_path, n_epochs=args.epochs,
        early_stop_patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing,
        ckpt_dir=args.ckpt_dir, run_tag=args.tag, verbose=args.verbose)
