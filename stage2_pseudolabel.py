"""
Stage 2 - BSD35k-CS as external data via PSEUDO-LABELING.

Why pseudo-labels (not the crowd labels): BSD35k-CS labels are noisy
(authors self-tagged). We IGNORE them and instead use our strong Stage-1
ensemble's high-confidence predictions as labels. So we use the 35k for its
AUDIO/TEXT embeddings, not its labels.

Pipeline:
  1. Load the 5 Stage-1 fold checkpoints (the 0.8226 ensemble).
  2. Run the ensemble over all 33,829 BSD35k embeddings -> soft predictions.
  3. Keep only HIGH-CONFIDENCE samples, with a PER-CLASS cap so the huge
     classes (fx, ss) don't drown the rare ones (is-w, sp-c).
  4. Add the kept pseudo-labeled 35k samples to the BSD10k DEV set.
  5. Retrain 5 folds on the enlarged dev set; evaluate the new ensemble on
     the SAME locked BSD10k test set (untouched) -> compare to 0.8226.

All on frozen CLAP features (same 630k-audioset-fusion-best checkpoint that
generated both datasets' embeddings) => fast.
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
# reuse the exact Stage-1 model so checkpoints load cleanly
from stage1_ensemble import Classifier, FocalLossWithSmoothing


# ──────────────────────────────────────────────────────────────────────
# Datasets keyed by an explicit features ROOT (so 10k and 35k both work)
# ──────────────────────────────────────────────────────────────────────
class EmbDataset(Dataset):
    """Reads concatenated audio+text emb for rows of a df, from a given
    features root. `label_col` gives the (pseudo or true) label as a class str."""
    def __init__(self, df, features_root, label_col="class"):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(features_root) / "clap_audio_embeddings"
        self.text_dir  = Path(features_root) / "clap_text_embeddings"
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a = np.load(self.audio_dir / f"{row.sound_id}.npy").astype(np.float32)
        t = np.load(self.text_dir  / f"{row.sound_id}.npy").astype(np.float32)
        emb = np.concatenate([a, t])
        return torch.tensor(emb), torch.tensor(CLS2IDX[row[self.label_col]], dtype=torch.long)


class MixedEmbDataset(Dataset):
    """Concatenates samples from possibly DIFFERENT feature roots (10k dev +
    35k pseudo). Each record carries its own features_root."""
    def __init__(self, records):
        # records: list of dicts {sound_id, class, features_root}
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        a = np.load(Path(r["features_root"]) / "clap_audio_embeddings" / f"{r['sound_id']}.npy").astype(np.float32)
        t = np.load(Path(r["features_root"]) / "clap_text_embeddings"  / f"{r['sound_id']}.npy").astype(np.float32)
        emb = np.concatenate([a, t])
        return torch.tensor(emb), torch.tensor(CLS2IDX[r["class"]], dtype=torch.long)


@torch.no_grad()
def ensemble_probs(models, df, features_root, device, batch_size=512):
    # manual batched loader (labels not needed for inference)
    audio_dir = Path(features_root) / "clap_audio_embeddings"
    text_dir  = Path(features_root) / "clap_text_embeddings"
    ids = df["sound_id"].tolist()
    batch, batch_ids, all_probs = [], [], []
    def flush():
        nonlocal batch, batch_ids
        if not batch:
            return
        x = torch.tensor(np.stack(batch)).to(device)
        p = torch.zeros(x.size(0), len(CLASSES), device=device)
        for m in models:
            p += F.softmax(m(x), dim=1)
        p /= len(models)
        all_probs.append(p.cpu())
        batch, batch_ids = [], []
    for sid in ids:
        a = np.load(audio_dir / f"{sid}.npy").astype(np.float32)
        t = np.load(text_dir  / f"{sid}.npy").astype(np.float32)
        batch.append(np.concatenate([a, t]))
        batch_ids.append(sid)
        if len(batch) == batch_size:
            flush()
    flush()
    return torch.cat(all_probs)  # [N, 23] in df order


def load_stage1_models(ckpt_dir, device, tag="stage1"):
    models = []
    for fold in range(1, 6):
        path = Path(ckpt_dir) / f"{tag}_fold{fold}.ckpt"
        m = Classifier.load_from_checkpoint(str(path), map_location=device).eval().to(device)
        models.append(m)
    return models


def select_pseudo(df35, probs, conf_thresh, per_class_cap):
    """Keep samples whose ensemble max-prob >= conf_thresh, capped per class."""
    conf, pred = probs.max(dim=1)
    conf = conf.numpy(); pred = pred.numpy()
    df = df35.copy().reset_index(drop=True)
    df["pseudo_conf"] = conf
    df["pseudo_class"] = [IDX2CLS[int(p)] for p in pred]
    df = df[df["pseudo_conf"] >= conf_thresh]
    # per-class cap: take the most confident up to cap per pseudo_class
    kept = []
    for cls, grp in df.groupby("pseudo_class"):
        kept.append(grp.sort_values("pseudo_conf", ascending=False).head(per_class_cap))
    out = pd.concat(kept).reset_index(drop=True) if kept else df.iloc[:0]
    return out


def train_eval(records_train, val_df, val_root, test_df, test_root, test_labels,
               device, n_epochs, patience, gamma, ls, ckpt_dir, fold):
    train_dl = DataLoader(MixedEmbDataset(records_train), batch_size=256,
                          shuffle=True, num_workers=4)
    val_dl   = DataLoader(EmbDataset(val_df, val_root), batch_size=256,
                          shuffle=False, num_workers=4)
    logger = WandbLogger(project="dcase2026-task1", name=f"stage2_fold{fold}")
    cbs = [
        EarlyStopping(monitor="val/hF_hier", patience=patience, mode="max"),
        ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"stage2_fold{fold}",
                        monitor="val/hF_hier", mode="max", save_top_k=1,
                        save_weights_only=True, enable_version_counter=False),
    ]
    trainer = pl.Trainer(max_epochs=n_epochs, logger=logger, accelerator="auto",
                         devices=1, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=cbs, log_every_n_steps=10)
    model = Classifier(gamma=gamma, label_smoothing=ls)
    trainer.fit(model, train_dl, val_dl)
    best = Classifier.load_from_checkpoint(cbs[1].best_model_path,
                                           map_location=device).eval().to(device)
    logger.experiment.finish()
    return best


def run(bsd10k_path, bsd35k_path, conf_thresh=0.9, per_class_cap=1500,
        n_epochs=50, patience=10, gamma=2.0, label_smoothing=0.1,
        stage1_ckpt="checkpoints_stage1", ckpt_dir="checkpoints_stage2"):
    pl.seed_everything(SEED, workers=True)
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f10 = str(Path(bsd10k_path) / "features")
    f35 = str(Path(bsd35k_path) / "features")

    # --- 10k splits (same locked test as every stage) ---
    df10 = load_metadata(bsd10k_path)
    test_idx, folds = get_splits(df10, seed=SEED)
    test_df = df10.iloc[test_idx].reset_index(drop=True)
    test_labels = [CLS2IDX[c] for c in test_df["class"]]

    # --- load Stage-1 ensemble & pseudo-label the 35k ---
    print("loading Stage-1 ensemble...")
    models = load_stage1_models(stage1_ckpt, device)

    df35 = pd.read_csv(Path(bsd35k_path) / "metadata" / "BSD35k-CS_metadata.csv")
    df35["sound_id"] = df35["sound_id"].astype(str).str.strip()
    df35 = df35[df35["class"].isin(CLASSES)].reset_index(drop=True)
    print(f"pseudo-labeling {len(df35)} BSD35k samples...")
    p35 = ensemble_probs(models, df35, f35, device)
    pseudo = select_pseudo(df35, p35, conf_thresh, per_class_cap)
    print(f"kept {len(pseudo)} pseudo-labeled samples "
          f"(conf>={conf_thresh}, cap={per_class_cap}/class)")
    print("pseudo class distribution:")
    print(pseudo["pseudo_class"].value_counts().to_string())

    # --- retrain 5 folds: 10k dev fold + pseudo-35k, eval on locked test ---
    test_probs = []
    cv = []
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n── stage2 fold {fold}/5 ──")
        dev_train = df10.iloc[tr_idx]
        val_df    = df10.iloc[va_idx].reset_index(drop=True)

        records = [{"sound_id": str(r.sound_id), "class": r["class"],
                    "features_root": f10} for _, r in dev_train.iterrows()]
        records += [{"sound_id": str(r.sound_id), "class": r["pseudo_class"],
                     "features_root": f35} for _, r in pseudo.iterrows()]
        print(f"  train = {len(dev_train)} real + {len(pseudo)} pseudo = {len(records)}")

        best = train_eval(records, val_df, f10, test_df, f10, test_labels,
                          device, n_epochs, patience, gamma, label_smoothing,
                          ckpt_dir, fold)
        tp = ensemble_probs([best], test_df, f10, device)
        test_probs.append(tp)
        _, _, hf = evaluate_probs(tp, test_labels, use_hierarchy=True)
        cv.append(hf)
        print(f"  fold {fold} on locked test: {hf:.4f}")

    ens = torch.stack(test_probs).mean(dim=0)
    _, _, ens_hf = evaluate_probs(ens, test_labels, use_hierarchy=True)

    print(f"\n{'='*60}")
    print(f"STAGE 2 — BSD35k PSEUDO-LABEL AUGMENTATION (locked test)")
    print(f"{'='*60}")
    print(f"  conf_thresh={conf_thresh}, per_class_cap={per_class_cap}")
    print(f"  mean of 5 single models:    {float(np.mean(cv)):.4f}")
    print(f"  5-model ensemble hF_hier:   {ens_hf:.4f}")
    print(f"{'-'*60}")
    print(f"  Stage-1 ensemble reference: 0.8226")
    print(f"  gain: {ens_hf - 0.8226:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsd10k_path", type=str, required=True)
    parser.add_argument("--bsd35k_path", type=str, required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.9)
    parser.add_argument("--per_class_cap", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints_stage1")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_stage2")
    args = parser.parse_args()
    run(args.bsd10k_path, args.bsd35k_path, conf_thresh=args.conf_thresh,
        per_class_cap=args.per_class_cap, n_epochs=args.epochs, patience=args.patience,
        gamma=args.gamma, label_smoothing=args.label_smoothing,
        stage1_ckpt=args.stage1_ckpt, ckpt_dir=args.ckpt_dir)
