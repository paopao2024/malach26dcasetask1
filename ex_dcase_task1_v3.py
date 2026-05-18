"""
DCASE 2026 Task 1 - v3
Adds: early stopping, clean summary table, quieter output, optional flags
for hierarchy-aware prediction and class-balanced loss.
"""

import os
import warnings
import logging

# silence the noise BEFORE importing torch/lightning
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
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from pathlib import Path
import argparse


# ── taxonomy ────────────────────────────────────────────────────────────
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


# ── hierarchical f-score ────────────────────────────────────────────────
def hierarchical_f(true_labels, pred_labels, lam=0.75):
    total = 1.0 + lam
    hp_per_class = defaultdict(list)
    hr_per_class = defaultdict(list)

    for true, pred in zip(true_labels, pred_labels):
        true_set = {true: 1.0, BST[true]: lam}
        pred_set = {pred: 1.0, BST[pred]: lam}
        overlap = sum(
            min(true_set.get(k, 0), pred_set.get(k, 0))
            for k in set(true_set) | set(pred_set)
        )
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


# ── dataset ─────────────────────────────────────────────────────────────
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
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 23),
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, class_weights=None, verbose=False):
        super().__init__()
        self.model = MLP()
        self.lr = lr
        self.verbose = verbose
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        self.val_outputs.append({
            "loss":   loss.cpu(),
            "logits": logits.cpu(),
            "labels": y.cpu(),
        })

    def on_validation_epoch_end(self):
        all_loss   = torch.cat([o["loss"]   for o in self.val_outputs])
        all_logits = torch.cat([o["logits"] for o in self.val_outputs])
        all_labels = torch.cat([o["labels"] for o in self.val_outputs])

        true_str = [IDX2CLS[l.item()] for l in all_labels]

        preds_orig = all_logits.argmax(dim=1)
        pred_orig_str = [IDX2CLS[p.item()] for p in preds_orig]
        hp_o, hr_o, hf_o = hierarchical_f(true_str, pred_orig_str)

        preds_hier = hierarchy_aware_predict(all_logits)
        pred_hier_str = [IDX2CLS[p.item()] for p in preds_hier]
        hp_h, hr_h, hf_h = hierarchical_f(true_str, pred_hier_str)

        self.log_dict({
            "val/loss":    all_loss.mean().item(),
            "val/hP":      hp_o,
            "val/hR":      hr_o,
            "val/hF":      hf_o,
            "val/hP_hier": hp_h,
            "val/hR_hier": hr_h,
            "val/hF_hier": hf_h,
        })

        # only print every 10 epochs, or always if verbose
        if self.verbose or (self.current_epoch + 1) % 10 == 0:
            print(f"    epoch {self.current_epoch+1:3d}  "
                  f"loss={all_loss.mean():.4f}  "
                  f"hF={hf_o:.4f}  hF_hier={hf_h:.4f}")

        self.val_outputs.clear()


# ── main ────────────────────────────────────────────────────────────────
def run(
    dataset_path,
    n_epochs=50,
    batch_size=256,
    seed=42,
    use_class_balanced=False,
    use_hierarchy_eval=True,
    early_stop_patience=10,
    verbose=False,
    run_tag="v3",
):
    pl.seed_everything(seed, workers=True)

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")
    print(f"config: class_balanced={use_class_balanced}, "
          f"hierarchy_eval={use_hierarchy_eval}, "
          f"early_stop_patience={early_stop_patience}, "
          f"max_epochs={n_epochs}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # collect per-fold results to print a clean summary at the end
    results = []  # list of dicts: {fold, hF, hF_hier, stop_epoch}

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class"]), 1):
        print(f"\n── fold {fold}/5 ──────────────────────────────")

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        # class weights (only used if use_class_balanced=True)
        weights = None
        if use_class_balanced:
            class_counts = train_df["class"].value_counts()
            weights = np.zeros(len(CLASSES))
            for cls in CLASSES:
                weights[CLS2IDX[cls]] = 1.0 / class_counts.get(cls, 1)
            weights = weights / weights.mean()
            print(f"  class weights: [{weights.min():.3f}, {weights.max():.3f}]")

        train_dl = DataLoader(
            SoundDataset(train_df, dataset_path),
            batch_size=batch_size, shuffle=True, num_workers=4,
        )
        val_dl = DataLoader(
            SoundDataset(val_df, dataset_path),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

        logger = WandbLogger(
            project="dcase2026-task1",
            name=f"{run_tag}_fold{fold}",
        )

        # early stopping: monitor val/hF (higher is better)
        # stops if no improvement for `patience` epochs
        monitor_metric = "val/hF_hier" if use_hierarchy_eval else "val/hF"
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stop_patience,
            mode="max",
            verbose=False,
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            logger=logger,
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,  # skip the model summary print each fold
            callbacks=[early_stop],
            log_every_n_steps=10,
        )

        model = Classifier(class_weights=weights, verbose=verbose)
        trainer.fit(model, train_dl, val_dl)

        hf_orig = trainer.callback_metrics.get("val/hF",      torch.tensor(0)).item()
        hf_hier = trainer.callback_metrics.get("val/hF_hier", torch.tensor(0)).item()
        stop_epoch = trainer.current_epoch + 1  # 1-indexed for humans

        results.append({
            "fold": fold,
            "hF": hf_orig,
            "hF_hier": hf_hier,
            "stop_epoch": stop_epoch,
        })

        print(f"  fold {fold} done after {stop_epoch} epochs   "
              f"hF={hf_orig:.4f}   hF_hier={hf_hier:.4f}")

        logger.experiment.finish()

    # ── summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY ({run_tag})")
    print(f"{'='*60}")
    print(f"{'Fold':<6}{'Stop@':<8}{'hF (argmax)':<15}{'hF (hier)':<15}")
    print(f"{'-'*44}")
    for r in results:
        print(f"{r['fold']:<6}{r['stop_epoch']:<8}{r['hF']:<15.4f}{r['hF_hier']:<15.4f}")
    print(f"{'-'*44}")
    mean_hf      = np.mean([r['hF']      for r in results])
    mean_hf_hier = np.mean([r['hF_hier'] for r in results])
    std_hf       = np.std ([r['hF']      for r in results])
    std_hf_hier  = np.std ([r['hF_hier'] for r in results])
    print(f"{'MEAN':<6}{'':<8}{mean_hf:<15.4f}{mean_hf_hier:<15.4f}")
    print(f"{'STD':<6}{'':<8}{std_hf:<15.4f}{std_hf_hier:<15.4f}")
    print(f"{'='*60}")
    print(f"baseline (v1) target ~0.632 | your baseline was 0.8013")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping patience (epochs without improvement)")
    parser.add_argument("--class_balanced", action="store_true",
                        help="use class-balanced loss (default: off, since v2 showed it hurt)")
    parser.add_argument("--no_hierarchy", action="store_true",
                        help="disable hierarchy-aware prediction (only original argmax)")
    parser.add_argument("--verbose", action="store_true",
                        help="print every epoch (default: every 10)")
    parser.add_argument("--tag", type=str, default="v3",
                        help="prefix for W&B run names")
    args = parser.parse_args()

    run(
        args.dataset_path,
        n_epochs=args.epochs,
        early_stop_patience=args.patience,
        use_class_balanced=args.class_balanced,
        use_hierarchy_eval=not args.no_hierarchy,
        verbose=args.verbose,
        run_tag=args.tag,
    )
