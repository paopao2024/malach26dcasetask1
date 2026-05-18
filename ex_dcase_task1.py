"""
DCASE 2026 Task 1 - Heterogeneous Audio Classification
Baseline: MLP on top of frozen LAION-CLAP embeddings
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
import argparse


# ── the 23 classes and which top-level group they belong to ──────────────
# this is just the taxonomy written out as a Python dict
BST = {
    "m-sp": "m",  "m-si": "m",  "m-m": "m",
    "is-p": "is", "is-s": "is", "is-w": "is", "is-k": "is", "is-e": "is",
    "sp-s": "sp", "sp-c": "sp", "sp-p": "sp",
    "fx-o": "fx", "fx-v": "fx", "fx-m": "fx", "fx-h": "fx",
    "fx-a": "fx", "fx-n": "fx", "fx-ex": "fx", "fx-el": "fx",
    "ss-n": "ss", "ss-i": "ss", "ss-u": "ss", "ss-s": "ss",
}

# sorted list of all 23 classes so we always use the same order
CLASSES = sorted(BST.keys())
# map class name to number (e.g. "fx-a" -> 4) and back
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


# ── hierarchical f-score ──────────────────────────────────────────────────
# normal accuracy doesn't care HOW wrong you are
# this metric gives partial credit if you at least got the top-level right
def hierarchical_f(true_labels, pred_labels, lam=0.75):
    total = 1.0 + lam  # max possible score per sample

    # collect scores grouped by class for macro averaging later
    hp_per_class = defaultdict(list)
    hr_per_class = defaultdict(list)

    for true, pred in zip(true_labels, pred_labels):
        # build weighted label sets: full credit for exact match, partial for parent
        true_set = {true: 1.0, BST[true]: lam}
        pred_set = {pred: 1.0, BST[pred]: lam}

        # how much do the two sets overlap?
        overlap = sum(
            min(true_set.get(k, 0), pred_set.get(k, 0))
            for k in set(true_set) | set(pred_set)
        )

        hp_per_class[pred].append(overlap / total)
        hr_per_class[true].append(overlap / total)

    # average per class, then average across all classes (macro)
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


# ── dataset ───────────────────────────────────────────────────────────────
class SoundDataset(Dataset):
    def __init__(self, df, dataset_path):
        self.df = df.reset_index(drop=True)
        # point to where the .npy files live based on the actual folder structure
        self.audio_dir = Path(dataset_path) / "features" / "clap_audio_embeddings"
        self.text_dir  = Path(dataset_path) / "features" / "clap_text_embeddings"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load the two .npy files for this sound and join them into one vector
        audio = np.load(self.audio_dir / f"{row.sound_id}.npy").astype(np.float32)
        text  = np.load(self.text_dir  / f"{row.sound_id}.npy").astype(np.float32)
        emb   = np.concatenate([audio, text])  # 512 + 512 = 1024 numbers

        label = CLS2IDX[row["class"]]

        return torch.tensor(emb), torch.tensor(label, dtype=torch.long)


# ── model ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 1024 inputs -> 256 -> 128 -> 23 outputs, dropout prevents overfitting
        self.layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 23),
        )

    def forward(self, x):
        return self.layers(x)  # raw scores, not probabilities


# ── pytorch lightning wraps the training loop so we don't write it manually
class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = MLP()
        self.lr = lr
        self.val_outputs = []  # store val results to compute hF at epoch end

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        preds = logits.argmax(dim=1)
        # save predictions and labels, process them all together at epoch end
        self.val_outputs.append({
            "loss":   loss.cpu(),
            "preds":  preds.cpu(),
            "labels": y.cpu(),
        })

    def on_validation_epoch_end(self):
        # gather everything from all validation batches
        all_loss   = torch.cat([o["loss"]   for o in self.val_outputs])
        all_preds  = torch.cat([o["preds"]  for o in self.val_outputs])
        all_labels = torch.cat([o["labels"] for o in self.val_outputs])

        # convert numbers back to class name strings for the hF function
        true = [IDX2CLS[l.item()] for l in all_labels]
        pred = [IDX2CLS[p.item()] for p in all_preds]

        hp, hr, hf = hierarchical_f(true, pred)

        # these get sent to W&B automatically
        self.log_dict({
            "val/loss": all_loss.mean().item(),
            "val/hP":   hp,
            "val/hR":   hr,
            "val/hF":   hf,
        })

        print(f"  epoch {self.current_epoch}  "
              f"loss={all_loss.mean():.4f}  hF={hf:.4f}")

        self.val_outputs.clear()


# ── main ──────────────────────────────────────────────────────────────────
def run(dataset_path, n_epochs=50, batch_size=256, seed=42):
    pl.seed_everything(seed)

    # load the csv, drop anything with a class we don't recognise
    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")

    # split into 5 folds, stratified so each fold has similar class balance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_hf = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class"]), 1):
        print(f"\n── fold {fold}/5 ──────────────────────────")

        train_dl = DataLoader(
            SoundDataset(df.iloc[train_idx], dataset_path),
            batch_size=batch_size, shuffle=True, num_workers=4,
        )
        val_dl = DataLoader(
            SoundDataset(df.iloc[val_idx], dataset_path),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

        # one W&B run per fold so you can compare them on the dashboard
        logger = WandbLogger(
            project="dcase2026-task1",
            name=f"baseline_fold{fold}",
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            logger=logger,
            accelerator="auto",       # GPU if available, CPU otherwise
            enable_progress_bar=False,
        )

        trainer.fit(Classifier(), train_dl, val_dl)

        best_hf = trainer.callback_metrics.get("val/hF", torch.tensor(0)).item()
        fold_hf.append(best_hf)
        print(f"  fold {fold} done  hF = {best_hf:.4f}")

        logger.experiment.finish()

    # final summary
    print(f"\n{'='*40}")
    print(f"mean hF = {np.mean(fold_hf):.4f}  (target ~0.632)")
    print(f"{'='*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="path to the 17250001 folder")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs)
