"""
DCASE 2026 Task 1 - v2
Baseline + Hierarchy-aware prediction + Class-balanced loss
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


# ── the 23 classes and their top-level parents ──────────────────────────
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

# top-level classes and their child indices (precomputed for speed)
TOP_LEVELS = sorted(set(BST.values()))  # ['fx', 'is', 'm', 'sp', 'ss']
TOP2IDX = {t: i for i, t in enumerate(TOP_LEVELS)}
TOP_TO_CHILDREN_IDX = {
    top: [CLS2IDX[c] for c in CLASSES if BST[c] == top]
    for top in TOP_LEVELS
}


# ── hierarchical f-score (unchanged from baseline) ──────────────────────
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


# ── NEW: hierarchy-aware prediction ─────────────────────────────────────
def hierarchy_aware_predict(logits):
    """
    Combine second-level probabilities with their parent's probability.
    Returns the index of the class that maximizes P(child) * P(parent).
    """
    probs = F.softmax(logits, dim=1)  # [B, 23]

    # P(top-level) = sum of P(its children)
    top_probs = torch.zeros(logits.size(0), len(TOP_LEVELS), device=logits.device)
    for top in TOP_LEVELS:
        children_idx = TOP_TO_CHILDREN_IDX[top]
        top_probs[:, TOP2IDX[top]] = probs[:, children_idx].sum(dim=1)

    # adjusted score = P(class) * P(its parent)
    adjusted = probs.clone()
    for i, cls in enumerate(CLASSES):
        parent_idx = TOP2IDX[BST[cls]]
        adjusted[:, i] = probs[:, i] * top_probs[:, parent_idx]

    return adjusted.argmax(dim=1)


# ── dataset (unchanged) ─────────────────────────────────────────────────
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


# ── model (unchanged) ───────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.layers(x)


# ── classifier with class weights + hierarchy-aware eval ────────────────
class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, class_weights=None):
        super().__init__()
        self.model = MLP()
        self.lr = lr
        # register as buffer so it moves to GPU automatically with the model
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
        # NEW: weighted cross-entropy (class-balanced)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        # store raw logits so we can compute BOTH metric variants
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

        # original argmax prediction
        preds_orig = all_logits.argmax(dim=1)
        pred_orig_str = [IDX2CLS[p.item()] for p in preds_orig]
        hp_o, hr_o, hf_o = hierarchical_f(true_str, pred_orig_str)

        # NEW: hierarchy-aware prediction
        preds_hier = hierarchy_aware_predict(all_logits)
        pred_hier_str = [IDX2CLS[p.item()] for p in preds_hier]
        hp_h, hr_h, hf_h = hierarchical_f(true_str, pred_hier_str)

        # log both, so we can compare
        self.log_dict({
            "val/loss":    all_loss.mean().item(),
            "val/hP":      hp_o,
            "val/hR":      hr_o,
            "val/hF":      hf_o,
            "val/hP_hier": hp_h,
            "val/hR_hier": hr_h,
            "val/hF_hier": hf_h,
        })

        print(f"  epoch {self.current_epoch}  loss={all_loss.mean():.4f}  "
              f"hF={hf_o:.4f}  hF_hier={hf_h:.4f}")

        self.val_outputs.clear()


# ── main ────────────────────────────────────────────────────────────────
def run(dataset_path, n_epochs=50, batch_size=256, seed=42):
    pl.seed_everything(seed)

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"loaded {len(df)} sounds, {df['class'].nunique()} classes")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_hf_orig = []
    fold_hf_hier = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class"]), 1):
        print(f"\n── fold {fold}/5 ──────────────────────────")

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        # NEW: compute class weights (inverse frequency, normalized)
        class_counts = train_df["class"].value_counts()
        weights = np.zeros(len(CLASSES))
        for cls in CLASSES:
            count = class_counts.get(cls, 1)
            weights[CLS2IDX[cls]] = 1.0 / count
        # normalize to mean 1 so the loss scale stays comparable
        weights = weights / weights.mean()
        print(f"  class weights range: [{weights.min():.3f}, {weights.max():.3f}]")

        train_dl = DataLoader(
            SoundDataset(train_df, dataset_path),
            batch_size=batch_size, shuffle=True, num_workers=4,
        )
        val_dl = DataLoader(
            SoundDataset(val_df, dataset_path),
            batch_size=batch_size, shuffle=False, num_workers=4,
        )

        # name v2 runs distinctly so they don't get mixed up with baseline in W&B
        logger = WandbLogger(
            project="dcase2026-task1",
            name=f"v2_fold{fold}",
        )

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            logger=logger,
            accelerator="auto",
            enable_progress_bar=False,
        )

        model = Classifier(class_weights=weights)
        trainer.fit(model, train_dl, val_dl)

        hf_orig = trainer.callback_metrics.get("val/hF",      torch.tensor(0)).item()
        hf_hier = trainer.callback_metrics.get("val/hF_hier", torch.tensor(0)).item()
        fold_hf_orig.append(hf_orig)
        fold_hf_hier.append(hf_hier)
        print(f"  fold {fold} done  hF={hf_orig:.4f}  hF_hier={hf_hier:.4f}")

        logger.experiment.finish()

    print(f"\n{'='*50}")
    print(f"mean hF (original argmax)    = {np.mean(fold_hf_orig):.4f}")
    print(f"mean hF (hierarchy-aware)    = {np.mean(fold_hf_hier):.4f}")
    print(f"baseline target ~0.632")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="path to the BSD10k folder (contains features/ and metadata/)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs)
