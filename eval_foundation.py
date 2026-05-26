"""
Evaluation foundation for DCASE 2026 Task 1 (shared by all stages).

Structure (decided: most defensible):
  - LOCKED TEST SET: a stratified 20% held out with seed 42. NO model ever
    trains on it. Every stage reports its final number on THIS set, so all
    numbers are directly comparable and leakage-free.
  - 5-FOLD CV on the remaining 80%: gives 5 diverse models (the ensemble) and
    a variance estimate. We develop/tune on CV, then confirm ONCE on the test.

This module exposes:
  - BST taxonomy, class maps, hierarchical_f, hierarchy_aware_predict  (canonical)
  - get_splits(df, seed)  -> (test_idx, list_of_5_(train_idx,val_idx))
  - load_metadata(dataset_path)
so stage scripts import from here instead of re-defining (prevents drift).

Running this file directly just prints the split sizes as a sanity check.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split


# ──────────────────────────────────────────────────────────────────────
# Canonical taxonomy + metric (single source of truth for every stage)
# ──────────────────────────────────────────────────────────────────────
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

TEST_FRACTION = 0.20
N_FOLDS = 5
SEED = 42


def hierarchical_f(true_labels, pred_labels, lam=0.75):
    """Macro hierarchical F (lambda=0.75). true/pred are lists of class STRINGS."""
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


def hierarchy_aware_predict(probs):
    """probs: torch tensor [N,23] of softmax probabilities. Returns argmax idx
    after re-weighting each class by its parent's aggregated probability."""
    import torch
    top_probs = torch.zeros(probs.size(0), len(TOP_LEVELS), device=probs.device)
    for top in TOP_LEVELS:
        children_idx = TOP_TO_CHILDREN_IDX[top]
        top_probs[:, TOP2IDX[top]] = probs[:, children_idx].sum(dim=1)
    adjusted = probs.clone()
    for i, cls in enumerate(CLASSES):
        parent_idx = TOP2IDX[BST[cls]]
        adjusted[:, i] = probs[:, i] * top_probs[:, parent_idx]
    return adjusted.argmax(dim=1)


def load_metadata(dataset_path):
    """Load + filter BSD10k metadata to the 23 BST classes, stable order."""
    df = pd.read_csv(Path(dataset_path) / "metadata" / "BSD10k_metadata.csv")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    return df


def get_splits(df, seed=SEED):
    """Return (test_idx, folds) where:
       - test_idx : np.array of row indices in the LOCKED test set (20%)
       - folds    : list of 5 (train_idx, val_idx) tuples over the other 80%
    All indices are positions into `df` (which must be reset_index'd).
    Deterministic given seed.
    """
    labels = df["class"].values
    all_idx = np.arange(len(df))

    # locked stratified test set
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_FRACTION, random_state=seed,
        stratify=labels,
    )
    dev_idx = np.sort(dev_idx)
    test_idx = np.sort(test_idx)

    # 5-fold CV within the development (80%) portion
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    dev_labels = labels[dev_idx]
    folds = []
    for tr_rel, va_rel in skf.split(dev_idx, dev_labels):
        folds.append((dev_idx[tr_rel], dev_idx[va_rel]))

    return test_idx, folds


def evaluate_probs(probs, true_idx, use_hierarchy=True):
    """probs: torch [N,23]; true_idx: list/array of int label indices.
    Returns (hp, hr, hf) using either hierarchy-aware or plain argmax."""
    import torch
    if use_hierarchy:
        pred = hierarchy_aware_predict(probs)
    else:
        pred = probs.argmax(dim=1)
    true_str = [IDX2CLS[int(t)] for t in true_idx]
    pred_str = [IDX2CLS[int(p)] for p in pred]
    return hierarchical_f(true_str, pred_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    df = load_metadata(args.dataset_path)
    test_idx, folds = get_splits(df)
    print(f"total samples (23 classes): {len(df)}")
    print(f"LOCKED test set:            {len(test_idx)}  ({TEST_FRACTION:.0%})")
    print(f"development set:            {len(df) - len(test_idx)}")
    print(f"{N_FOLDS}-fold CV on development:")
    for i, (tr, va) in enumerate(folds, 1):
        print(f"  fold {i}: train={len(tr):5d}  val={len(va):5d}")
    # sanity: test never overlaps any fold's train/val
    test_set = set(test_idx.tolist())
    leak = any(test_set & set(tr.tolist()) or test_set & set(va.tolist())
               for tr, va in folds)
    print(f"\nleakage check (test vs all folds): {'LEAK!' if leak else 'clean ✓'}")
    # class coverage in test
    import collections
    cc = collections.Counter(df.iloc[test_idx]["class"])
    print(f"test covers {len(cc)}/23 classes; min class count = {min(cc.values())}")
