"""
Stage 1 - Ensemble the 5 v6 fold checkpoints.

Two ensemble views, both reported:

1) OUT-OF-FOLD (honest, no leakage): each sample is predicted by the ONE model
   whose training set did NOT contain it (its own fold's model). This reproduces
   the per-fold CV number but lets us also form the soft-vote below.

2) FULL ENSEMBLE soft-vote: average the softmax probabilities of ALL 5 models
   for every sample, then predict. This is the actual "ensemble" and is what
   usually beats the single-fold mean. NOTE: 4 of the 5 models saw each sample
   in training, so this number is mildly optimistic on the train portion --
   but it is the standard way ensembles are reported and is what you would
   submit (at test time NONE of the models have seen the eval clips, so the
   soft-vote is exactly the deployment setup).

Run AFTER ex_dcase_task1_v6_ckpt.py has produced checkpoints/v6_fold{1..5}.ckpt
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import argparse

# reuse everything from the training script
from ex_dcase_task1_v6_ckpt import (
    Classifier, SoundDataset, CLASSES, CLS2IDX, IDX2CLS,
    hierarchical_f, hierarchy_aware_predict, BST,
)


def load_fold_models(ckpt_dir, device, tag="v6"):
    models = []
    for fold in range(1, 6):
        path = Path(ckpt_dir) / f"{tag}_fold{fold}.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        m = Classifier.load_from_checkpoint(str(path), map_location=device)
        m.eval().to(device)
        models.append(m)
    print(f"loaded {len(models)} fold models from {ckpt_dir}")
    return models


@torch.no_grad()
def logits_for(model, df, dataset_path, device, batch_size=256):
    """Return softmax probs [N, 23] from one model over the given df, in df order."""
    dl = DataLoader(SoundDataset(df, dataset_path), batch_size=batch_size,
                    shuffle=False, num_workers=4)
    out = []
    for x, _ in dl:
        out.append(F.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(out)


def evaluate(true_str, pred_idx):
    pred_str = [IDX2CLS[int(p)] for p in pred_idx]
    return hierarchical_f(true_str, pred_str)


def run(dataset_path, ckpt_dir="checkpoints", seed=42, tag="v6"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    labels = df["class"].tolist()
    true_str_all = labels[:]  # already class strings

    models = load_fold_models(ckpt_dir, device, tag=tag)

    # Recreate the SAME folds used in training (seed 42) to know which model
    # is the out-of-fold model for each sample.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_of = np.full(len(df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["class"])):
        fold_of[val_idx] = fold  # 0-based; model index in `models`

    # ---- compute every model's probs over the FULL dataset once ----
    all_probs = []  # list of [N,23] tensors, one per model
    for i, m in enumerate(models, 1):
        all_probs.append(logits_for(m, df, dataset_path, device))
        print(f"  scored model {i}/5")
    all_probs = torch.stack(all_probs)  # [5, N, 23]

    N = all_probs.shape[1]

    # ---- (1) OUT-OF-FOLD prediction: each sample uses its own fold's model ----
    oof_probs = torch.zeros(N, len(CLASSES))
    for n in range(N):
        oof_probs[n] = all_probs[fold_of[n], n]
    oof_argmax = oof_probs.argmax(dim=1)
    oof_hier   = hierarchy_aware_predict(oof_probs)
    hp1, hr1, hf1_arg  = evaluate(true_str_all, oof_argmax)
    _,   _,   hf1_hier = evaluate(true_str_all, oof_hier)

    # ---- (2) FULL soft-vote ensemble: average all 5 models ----
    ens_probs  = all_probs.mean(dim=0)  # [N,23]
    ens_argmax = ens_probs.argmax(dim=1)
    ens_hier   = hierarchy_aware_predict(ens_probs)
    hp2, hr2, hf2_arg  = evaluate(true_str_all, ens_argmax)
    _,   _,   hf2_hier = evaluate(true_str_all, ens_hier)

    print(f"\n{'='*64}")
    print(f"STAGE 1 — v6 ENSEMBLE")
    print(f"{'='*64}")
    print(f"{'view':<34}{'hF (argmax)':<16}{'hF (hier)':<16}")
    print(f"{'-'*64}")
    print(f"{'out-of-fold (honest CV)':<34}{hf1_arg:<16.4f}{hf1_hier:<16.4f}")
    print(f"{'5-model soft-vote ensemble':<34}{hf2_arg:<16.4f}{hf2_hier:<16.4f}")
    print(f"{'-'*64}")
    print(f"single-fold v6 reference: ~0.8048")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="v6")
    args = parser.parse_args()
    run(args.dataset_path, ckpt_dir=args.ckpt_dir, seed=args.seed, tag=args.tag)
