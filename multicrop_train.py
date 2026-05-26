"""
Multi-crop training - train on WINDOWS, predict per CLIP.

Each clip was sliced into overlapping 7s windows (extract_windows.py), giving
34,745 windows from 10,956 clips (3.17x more examples) plus temporal detail the
single pooled embedding loses.

Training:  each WINDOW is a training example, labeled with its parent clip's
           class. Audio = the window's CLAP embedding; text = the clip's text
           embedding (broadcast to all its windows).
Inference: for each test CLIP, run all its windows, AVERAGE their softmax
           probabilities -> one clip-level prediction. Evaluated on the same
           locked BSD10k test set, comparable to the 0.8226 Stage-1 ensemble.

5-fold ensemble, splits from eval_foundation. CRITICAL: folds are made at the
CLIP level (all windows of a clip stay together) to avoid leakage between
train and val/test. Frozen features => fast.
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
    CLASSES, CLS2IDX, IDX2CLS,
    load_metadata, get_splits, evaluate_probs, SEED,
)
from stage1_ensemble import Classifier  # same MLP/loss/training as Stage-1


# ──────────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """One item per WINDOW. audio = window emb, text = clip's text emb."""
    def __init__(self, window_rows, win_dir, text_dir):
        # window_rows: list of dicts {sound_id, win_k, class}
        self.rows = window_rows
        self.win_dir = Path(win_dir)
        self.text_dir = Path(text_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        a = np.load(self.win_dir / f"{r['sound_id']}__w{r['win_k']}.npy").astype(np.float32)
        t = np.load(self.text_dir / f"{r['sound_id']}.npy").astype(np.float32)
        emb = np.concatenate([a, t])
        return torch.tensor(emb), torch.tensor(CLS2IDX[r["class"]], dtype=torch.long)


def build_window_rows(clip_df, win_index):
    """Expand a clip-level df into window-level rows using the window index."""
    nwin = dict(zip(win_index["sound_id"].astype(str), win_index["n_windows"]))
    rows = []
    for _, c in clip_df.iterrows():
        sid = str(c.sound_id)
        for k in range(int(nwin.get(sid, 1))):
            rows.append({"sound_id": sid, "win_k": k, "class": c["class"]})
    return rows


@torch.no_grad()
def clip_probs(model, clip_df, win_index, win_dir, text_dir, device, batch_size=512):
    """For each clip, average its windows' softmax -> [Nclips, 23] in clip_df order."""
    nwin = dict(zip(win_index["sound_id"].astype(str), win_index["n_windows"]))
    win_dir = Path(win_dir); text_dir = Path(text_dir)
    out = torch.zeros(len(clip_df), len(CLASSES))
    # flatten all windows, remember which clip each belongs to
    flat, owner = [], []
    for i, c in enumerate(clip_df.itertuples()):
        sid = str(c.sound_id)
        t = np.load(text_dir / f"{sid}.npy").astype(np.float32)
        for k in range(int(nwin.get(sid, 1))):
            a = np.load(win_dir / f"{sid}__w{k}.npy").astype(np.float32)
            flat.append(np.concatenate([a, t])); owner.append(i)
    counts = torch.zeros(len(clip_df))
    for start in range(0, len(flat), batch_size):
        xb = torch.tensor(np.stack(flat[start:start+batch_size])).to(device)
        pb = F.softmax(model(xb), dim=1).cpu()
        ob = owner[start:start+batch_size]
        for j, o in enumerate(ob):
            out[o] += pb[j]; counts[o] += 1
    out /= counts.clamp(min=1).unsqueeze(1)
    return out


def run(dataset_path, n_epochs=50, patience=10, gamma=2.0, label_smoothing=0.1,
        ckpt_dir="checkpoints_multicrop"):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat = Path(dataset_path) / "features"
    win_dir = feat / "clap_audio_win"
    text_dir = feat / "clap_text_embeddings"
    win_index = pd.read_csv(feat / "window_index.csv")
    win_index["sound_id"] = win_index["sound_id"].astype(str)

    df = load_metadata(dataset_path)
    df["sound_id"] = df["sound_id"].astype(str)
    test_idx, folds = get_splits(df, seed=SEED)         # CLIP-level splits
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_labels = [CLS2IDX[c] for c in test_df["class"]]

    test_probs, cv = [], []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n── multicrop fold {fold}/5 ──")
        train_clips = df.iloc[tr_idx]
        val_clips   = df.iloc[va_idx].reset_index(drop=True)

        train_rows = build_window_rows(train_clips, win_index)
        val_rows   = build_window_rows(val_clips, win_index)
        print(f"  train windows: {len(train_rows)} (from {len(train_clips)} clips)")

        train_dl = DataLoader(WindowDataset(train_rows, win_dir, text_dir),
                              batch_size=256, shuffle=True, num_workers=4)
        # validate at WINDOW level during training (cheap proxy); final number
        # below is the proper CLIP-level aggregated metric on the test set.
        val_dl = DataLoader(WindowDataset(val_rows, win_dir, text_dir),
                            batch_size=256, shuffle=False, num_workers=4)

        logger = WandbLogger(project="dcase2026-task1", name=f"multicrop_fold{fold}")
        cbs = [
            EarlyStopping(monitor="val/hF_hier", patience=patience, mode="max"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"multicrop_fold{fold}",
                            monitor="val/hF_hier", mode="max", save_top_k=1,
                            save_weights_only=True, enable_version_counter=False),
        ]
        trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="auto",
                             devices=1, enable_progress_bar=False,
                             enable_model_summary=False, callbacks=cbs, log_every_n_steps=20)
        model = Classifier(gamma=gamma, label_smoothing=label_smoothing)
        trainer.fit(model, train_dl, val_dl)

        best = Classifier.load_from_checkpoint(cbs[1].best_model_path,
                                               map_location=device).eval().to(device)
        # CLIP-level aggregated prediction on locked test
        tp = clip_probs(best, test_df, win_index, win_dir, text_dir, device)
        test_probs.append(tp)
        _, _, hf = evaluate_probs(tp, test_labels, use_hierarchy=True)
        cv.append(hf)
        print(f"  fold {fold} CLIP-level on locked test: {hf:.4f}")
        logger.experiment.finish()

    ens = torch.stack(test_probs).mean(dim=0)
    _, _, ens_hf = evaluate_probs(ens, test_labels, use_hierarchy=True)
    print(f"\n{'='*60}")
    print(f"MULTI-CROP (window train, clip-aggregate inference) — locked test")
    print(f"{'='*60}")
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
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_multicrop")
    args = parser.parse_args()
    run(args.dataset_path, n_epochs=args.epochs, patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing, ckpt_dir=args.ckpt_dir)
