"""
DCASE 2026 Task 1 - v7
Same ensemble framework as v5, with PANNs CNN14 added to the supported features.

Examples:
  --models clap panns              # 2-way ensemble: CLAP + PANNs
  --models clap passt panns        # 3-way ensemble: CLAP + PaSST + PANNs
  --models panns                   # standalone PANNs classifier (baseline check)
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
from pytorch_lightning.callbacks import EarlyStopping
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

FEATURE_CONFIGS = {
    "clap":            {"dirs": ["clap_audio_embeddings", "clap_text_embeddings"], "dim": 1024},
    "passt":           {"dirs": ["passt_embeddings"], "dim": 768},
    "panns":           {"dirs": ["panns_embeddings"], "dim": 2048},
    "passt+clap_text": {"dirs": ["passt_embeddings", "clap_text_embeddings"], "dim": 1280},
    "panns+clap_text": {"dirs": ["panns_embeddings", "clap_text_embeddings"], "dim": 2560},
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


def hierarchy_aware_from_probs(probs):
    top_probs = torch.zeros(probs.size(0), len(TOP_LEVELS))
    for top in TOP_LEVELS:
        children_idx = TOP_TO_CHILDREN_IDX[top]
        top_probs[:, TOP2IDX[top]] = probs[:, children_idx].sum(dim=1)
    adjusted = probs.clone()
    for i, cls in enumerate(CLASSES):
        parent_idx = TOP2IDX[BST[cls]]
        adjusted[:, i] = probs[:, i] * top_probs[:, parent_idx]
    return adjusted


class SoundDataset(Dataset):
    def __init__(self, df, feature_dirs):
        self.df = df.reset_index(drop=True)
        self.feature_dirs = [Path(d) for d in feature_dirs]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parts = []
        for fdir in self.feature_dirs:
            arr = np.load(fdir / f"{row.sound_id}.npy").astype(np.float32)
            parts.append(arr)
        emb = np.concatenate(parts)
        label = CLS2IDX[row["class"]]
        return torch.tensor(emb), torch.tensor(label, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 23),
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.model = MLP(input_dim)
        self.lr = lr
        self.val_outputs = []

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
        self.val_outputs.append({"logits": logits.cpu(), "labels": y.cpu()})

    def on_validation_epoch_end(self):
        all_logits = torch.cat([o["logits"] for o in self.val_outputs])
        all_labels = torch.cat([o["labels"] for o in self.val_outputs])
        true_str = [IDX2CLS[l.item()] for l in all_labels]
        preds = all_logits.argmax(dim=1)
        pred_str = [IDX2CLS[p.item()] for p in preds]
        _, _, hf = hierarchical_f(true_str, pred_str)
        self.log("val/hF", hf)
        self.val_outputs.clear()


def train_and_predict(features, train_df, val_df, dataset_path,
                      n_epochs, patience, batch_size, fold_num, run_tag):
    cfg = FEATURE_CONFIGS[features]
    input_dim = cfg["dim"]
    feature_dirs = [Path(dataset_path) / "features" / d for d in cfg["dirs"]]

    train_ds = SoundDataset(train_df, feature_dirs)
    val_ds   = SoundDataset(val_df,   feature_dirs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    logger = WandbLogger(project="dcase2026-task1",
                         name=f"{run_tag}_{features}_fold{fold_num}")
    early_stop = EarlyStopping(monitor="val/hF", patience=patience, mode="max", verbose=False)

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early_stop],
        log_every_n_steps=10,
    )
    model = Classifier(input_dim=input_dim)
    trainer.fit(model, train_dl, val_dl)
    logger.experiment.finish()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    probs_chunks, labels_chunks = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu()
            probs_chunks.append(probs)
            labels_chunks.append(y)

    return torch.cat(probs_chunks), torch.cat(labels_chunks)


def run_ensemble(dataset_path, feature_sets, n_epochs=50, patience=10,
                 batch_size=256, seed=42, run_tag="v7"):
    pl.seed_everything(seed, workers=True)

    for fs in feature_sets:
        if fs not in FEATURE_CONFIGS:
            raise ValueError(f"unknown features '{fs}'. options: {list(FEATURE_CONFIGS.keys())}")

    all_dirs = set()
    for fs in feature_sets:
        for d in FEATURE_CONFIGS[fs]["dirs"]:
            all_dirs.add(d)
    all_paths = [Path(dataset_path) / "features" / d for d in all_dirs]

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    keep = [all((d / f"{row.sound_id}.npy").exists() for d in all_paths)
            for _, row in df.iterrows()]
    df = df[keep].reset_index(drop=True)

    print(f"ensemble of: {feature_sets}")
    print(f"sounds with all required features: {len(df)}")
    print(f"max epochs: {n_epochs}, patience: {patience}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_results = {fs: [] for fs in feature_sets}
    fold_results["ensemble"]      = []
    fold_results["ensemble_hier"] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class"]), 1):
        print(f"══ fold {fold}/5 ═══════════════════════════════")
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        per_model_probs = []
        val_labels_ref = None

        for fs in feature_sets:
            print(f"  training {fs}...")
            probs, labels = train_and_predict(
                fs, train_df, val_df, dataset_path,
                n_epochs=n_epochs, patience=patience, batch_size=batch_size,
                fold_num=fold, run_tag=run_tag,
            )
            per_model_probs.append(probs)
            if val_labels_ref is None:
                val_labels_ref = labels

            preds = probs.argmax(dim=1)
            true_str = [IDX2CLS[l.item()] for l in labels]
            pred_str = [IDX2CLS[p.item()] for p in preds]
            _, _, hf = hierarchical_f(true_str, pred_str)
            fold_results[fs].append(hf)
            print(f"    {fs:30s}  hF = {hf:.4f}")

        ensemble_probs = torch.stack(per_model_probs).mean(dim=0)

        ens_preds = ensemble_probs.argmax(dim=1)
        true_str = [IDX2CLS[l.item()] for l in val_labels_ref]
        pred_str = [IDX2CLS[p.item()] for p in ens_preds]
        _, _, hf_ens = hierarchical_f(true_str, pred_str)
        fold_results["ensemble"].append(hf_ens)

        ens_probs_hier = hierarchy_aware_from_probs(ensemble_probs)
        ens_preds_hier = ens_probs_hier.argmax(dim=1)
        pred_str_hier = [IDX2CLS[p.item()] for p in ens_preds_hier]
        _, _, hf_ens_hier = hierarchical_f(true_str, pred_str_hier)
        fold_results["ensemble_hier"].append(hf_ens_hier)

        print(f"    {'ENSEMBLE (argmax)':30s}  hF = {hf_ens:.4f}")
        print(f"    {'ENSEMBLE (hierarchy-aware)':30s}  hF = {hf_ens_hier:.4f}\n")

    print(f"{'='*70}")
    print(f"SUMMARY ({run_tag})")
    print(f"ensemble of: {feature_sets}")
    print(f"{'='*70}\n")

    print("Individual models (mean ± std across folds):")
    for fs in feature_sets:
        scores = fold_results[fs]
        print(f"  {fs:30s}  hF = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    print("\nEnsemble:")
    ens = fold_results["ensemble"]
    ens_h = fold_results["ensemble_hier"]
    print(f"  argmax                          hF = {np.mean(ens):.4f} ± {np.std(ens):.4f}")
    print(f"  hierarchy-aware                 hF = {np.mean(ens_h):.4f} ± {np.std(ens_h):.4f}")

    print(f"\n{'='*70}")
    print(f"v1 baseline   : 0.8013")
    print(f"v3 default    : 0.8005")
    print(f"v5 combo      : 0.8002")
    print(f"v6 (classifier improvements) : 0.8048")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=["clap", "panns"],
                        help="feature sets to ensemble (space-separated)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tag", type=str, default="v7")
    args = parser.parse_args()

    run_ensemble(
        args.dataset_path,
        feature_sets=args.models,
        n_epochs=args.epochs,
        patience=args.patience,
        run_tag=args.tag,
    )
