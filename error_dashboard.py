"""
Error analysis dashboard for DCASE 2026 Task 1 — Stage-1 ensemble.

Reuses eval_foundation (taxonomy, metric, splits) and the trained Stage-1
checkpoints so every number here reconciles EXACTLY with the 0.8226 you
already reported. No model is trained or re-tuned. The locked test set is
analyzed in read-only fashion.

What it answers (the only question that matters before more modeling):
  Are the remaining errors FIXABLE (calibration / class-bias / imbalance)
  or INTRINSIC (label ambiguity, cross-top confusion)?

Outputs (all on the LOCKED TEST set, using the SAME hierarchy-aware
predictions that produce the 0.8226):
  1. Per-class table: support, hP, hR, per-class F, plain acc, top-level acc
  2. Top-level confusion matrix (5x5)  -- the expensive errors live here
  3. Sub-class confusion matrix (23x23, printed compactly + saved to CSV)
  4. The four error buckets from the diagnosis doc:
        E1 = wrong sub-class, correct top-level   (cheap: lam=0.75 partial credit)
        E2 = wrong top-level                       (expensive: full penalty)
        E3 = high-confidence wrong  (conf >= --hi_conf)
        E4 = low-confidence correct (conf <= --lo_conf)
  5. The headline read: how much of the total hF shortfall is E1 vs E2,
     i.e. is the loss "within-family" (fixable) or "cross-family" (intrinsic).

Usage:
  python error_dashboard.py --dataset_path /path/to/BSD10k --ckpt_dir checkpoints_stage1

Run stage1_ensemble.py FIRST so the 5 fold checkpoints exist.
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eval_foundation import (
    CLASSES, CLS2IDX, IDX2CLS, BST,
    TOP_LEVELS, TOP2IDX,
    hierarchical_f, hierarchy_aware_predict,
    load_metadata, get_splits, evaluate_probs, SEED,
)

# Reuse the EXACT model/dataset definitions from stage1 so checkpoints load
# cleanly and inference matches the reported pipeline byte-for-byte.
from stage1_ensemble import Classifier, SoundDataset, probs_on


# ──────────────────────────────────────────────────────────────────────
# Load the 5 Stage-1 fold checkpoints and rebuild the soft-vote on the test set
# ──────────────────────────────────────────────────────────────────────
def load_ensemble_test_probs(dataset_path, ckpt_dir, device):
    """Returns (ens_probs [Ntest,23], per_model_probs [5,Ntest,23], test_df)."""
    df = load_metadata(dataset_path)
    test_idx, _ = get_splits(df, seed=SEED)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    ckpt_dir = Path(ckpt_dir)
    per_model = []
    for fold in range(1, 6):
        ckpt = ckpt_dir / f"stage1_fold{fold}.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {ckpt}\n"
                f"Run stage1_ensemble.py first to generate the 5 fold models."
            )
        model = Classifier.load_from_checkpoint(str(ckpt), map_location=device).eval().to(device)
        per_model.append(probs_on(model, test_df, dataset_path, device))

    stacked = torch.stack(per_model)        # [5, Ntest, 23]
    ens = stacked.mean(dim=0)               # soft-vote (the 0.8226 path)
    return ens, stacked, test_df


# ──────────────────────────────────────────────────────────────────────
# Per-class diagnostics
# ──────────────────────────────────────────────────────────────────────
def per_class_table(true_idx, pred_idx):
    """Per-class hP, hR, F (decomposition), plain accuracy, top-level accuracy.

    NOTE on per-class F: the global metric computes ONE F from macro hP & hR.
    Re-slicing a single F per class is not the global number; we report
    per-class hP and hR (which ARE the genuine building blocks) plus a derived
    per-class F purely as a sortable diagnostic. Treat F as "where is this
    class weak", not as a re-decomposition of 0.8226.
    """
    true_str = [IDX2CLS[int(t)] for t in true_idx]
    pred_str = [IDX2CLS[int(p)] for p in pred_idx]
    lam = 0.75
    total = 1.0 + lam

    hp_per = defaultdict(list)
    hr_per = defaultdict(list)
    for t, p in zip(true_str, pred_str):
        t_set = {t: 1.0, BST[t]: lam}
        p_set = {p: 1.0, BST[p]: lam}
        overlap = sum(min(t_set.get(k, 0), p_set.get(k, 0))
                      for k in set(t_set) | set(p_set))
        hp_per[p].append(overlap / total)
        hr_per[t].append(overlap / total)

    support = Counter(true_str)
    rows = []
    for c in CLASSES:
        sup = support.get(c, 0)
        hp = float(np.mean(hp_per[c])) if hp_per[c] else 0.0
        hr = float(np.mean(hr_per[c])) if hr_per[c] else 0.0
        f = 2 * hp * hr / (hp + hr) if (hp + hr) > 0 else 0.0
        # plain + top-level accuracy on samples whose TRUE class is c
        correct = sum(1 for t, p in zip(true_str, pred_str) if t == c and p == c)
        top_correct = sum(1 for t, p in zip(true_str, pred_str)
                          if t == c and BST[p] == BST[c])
        acc = correct / sup if sup else 0.0
        top_acc = top_correct / sup if sup else 0.0
        rows.append({
            "class": c, "top": BST[c], "support": sup,
            "hP": round(hp, 4), "hR": round(hr, 4), "F": round(f, 4),
            "acc": round(acc, 4), "top_acc": round(top_acc, 4),
        })
    return pd.DataFrame(rows)


def top_level_confusion(true_idx, pred_idx):
    """5x5 top-level confusion (rows = true top, cols = pred top), counts."""
    mat = pd.DataFrame(0, index=TOP_LEVELS, columns=TOP_LEVELS, dtype=int)
    for t, p in zip(true_idx, pred_idx):
        tt = BST[IDX2CLS[int(t)]]
        pp = BST[IDX2CLS[int(p)]]
        mat.loc[tt, pp] += 1
    return mat


def sub_confusion(true_idx, pred_idx):
    """23x23 sub-class confusion (rows = true, cols = pred), counts."""
    mat = pd.DataFrame(0, index=CLASSES, columns=CLASSES, dtype=int)
    for t, p in zip(true_idx, pred_idx):
        mat.loc[IDX2CLS[int(t)], IDX2CLS[int(p)]] += 1
    return mat


# ──────────────────────────────────────────────────────────────────────
# The four error buckets + the headline E1-vs-E2 read
# ──────────────────────────────────────────────────────────────────────
def error_buckets(true_idx, pred_idx, conf, test_df, hi_conf=0.90, lo_conf=0.50):
    sound_ids = test_df["sound_id"].tolist()
    E1, E2, E3, E4 = [], [], [], []  # each: (sound_id, true, pred, conf)

    for sid, t, p, c in zip(sound_ids, true_idx, pred_idx, conf):
        t, p, c = int(t), int(p), float(c)
        tc, pc = IDX2CLS[t], IDX2CLS[p]
        rec = (sid, tc, pc, round(c, 3))
        wrong = (t != p)
        if wrong and BST[tc] == BST[pc]:
            E1.append(rec)                       # wrong sub, right top
        if wrong and BST[tc] != BST[pc]:
            E2.append(rec)                       # wrong top
        if wrong and c >= hi_conf:
            E3.append(rec)                       # confidently wrong
        if (not wrong) and c <= lo_conf:
            E4.append(rec)                       # correct but unsure
    return {"E1": E1, "E2": E2, "E3": E3, "E4": E4}


def hf_shortfall_attribution(true_idx, pred_idx):
    """Decompose the gap to a perfect score (hF=1) into the credit lost on
    E1 errors (each loses 1 - lam/(1+lam) of a unit) vs E2 errors (lose all).

    Per matched pair, overlap/total credit is:
      correct      -> 1.0
      E1 (wrong sub, right top) -> lam / (1+lam)         = 0.4286 with lam=0.75
      E2 (wrong top)            -> 0.0
    So credit LOST per error: E1 loses (1 - lam/(1+lam)); E2 loses 1.0.
    This is an instance-level attribution (not the macro-F itself) and is the
    cleanest way to see whether the shortfall is 'within-family' or 'cross-family'.
    """
    lam = 0.75
    total = 1.0 + lam
    e1_credit_each = lam / total           # credit an E1 error still earns
    n = len(true_idx)
    n_e1 = sum(1 for t, p in zip(true_idx, pred_idx)
               if t != p and BST[IDX2CLS[int(t)]] == BST[IDX2CLS[int(p)]])
    n_e2 = sum(1 for t, p in zip(true_idx, pred_idx)
               if t != p and BST[IDX2CLS[int(t)]] != BST[IDX2CLS[int(p)]])
    n_correct = n - n_e1 - n_e2

    lost_e1 = n_e1 * (1.0 - e1_credit_each)   # each E1 loses (1 - 0.4286)
    lost_e2 = n_e2 * 1.0                       # each E2 loses everything
    lost_total = lost_e1 + lost_e2
    return {
        "n": n, "n_correct": n_correct, "n_E1": n_e1, "n_E2": n_e2,
        "credit_lost_E1": lost_e1, "credit_lost_E2": lost_e2,
        "credit_lost_total": lost_total,
        "frac_loss_from_E1": (lost_e1 / lost_total) if lost_total else 0.0,
        "frac_loss_from_E2": (lost_e2 / lost_total) if lost_total else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints_stage1")
    ap.add_argument("--hi_conf", type=float, default=0.90)
    ap.add_argument("--lo_conf", type=float, default=0.50)
    ap.add_argument("--out_dir", type=str, default="error_analysis")
    ap.add_argument("--top_k_examples", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Stage-1 ensemble and scoring the locked test set...")
    ens_probs, _, test_df = load_ensemble_test_probs(args.dataset_path, args.ckpt_dir, device)
    true_idx = [CLS2IDX[c] for c in test_df["class"]]

    # SAME hierarchy-aware prediction path that produces the 0.8226 headline
    pred = hierarchy_aware_predict(ens_probs)
    pred_idx = pred.tolist()
    conf = ens_probs.max(dim=1).values.tolist()   # max softmax prob as confidence

    # confirm we reproduce the headline number before trusting any breakdown
    _, _, hf_hier = evaluate_probs(ens_probs, true_idx, use_hierarchy=True)
    _, _, hf_arg = evaluate_probs(ens_probs, true_idx, use_hierarchy=False)
    print(f"\nReconciliation — ensemble on locked test:")
    print(f"  hF (hierarchy-aware, the headline path): {hf_hier:.4f}")
    print(f"  hF (plain argmax):                       {hf_arg:.4f}")
    print(f"  (this should match your reported ~0.8226)\n")

    # 1. per-class table -------------------------------------------------
    tbl = per_class_table(true_idx, pred_idx).sort_values("F").reset_index(drop=True)
    tbl.to_csv(out / "per_class.csv", index=False)
    print("=" * 78)
    print("PER-CLASS DIAGNOSTICS (sorted worst F first)")
    print("=" * 78)
    print(tbl.to_string(index=False))

    # 2. top-level confusion --------------------------------------------
    topcm = top_level_confusion(true_idx, pred_idx)
    topcm.to_csv(out / "top_confusion.csv")
    print("\n" + "=" * 78)
    print("TOP-LEVEL CONFUSION  (rows=true, cols=pred)  -- cross-cell = expensive E2")
    print("=" * 78)
    print(topcm.to_string())
    off = topcm.values.sum() - np.trace(topcm.values)
    print(f"\n  off-diagonal (E2 cross-top errors): {off} of {topcm.values.sum()} test clips")

    # 3. sub confusion (saved; print only the worst confusions) ----------
    subcm = sub_confusion(true_idx, pred_idx)
    subcm.to_csv(out / "sub_confusion.csv")
    confusions = []
    for t in CLASSES:
        for p in CLASSES:
            if t != p and subcm.loc[t, p] > 0:
                confusions.append((subcm.loc[t, p], t, p, BST[t] == BST[p]))
    confusions.sort(reverse=True)
    print("\n" + "=" * 78)
    print("TOP SUB-CLASS CONFUSIONS (count | true -> pred | within-family?)")
    print("=" * 78)
    for cnt, t, p, same in confusions[:20]:
        tag = "within-top" if same else "CROSS-TOP"
        print(f"  {cnt:4d}  {t:6s} -> {p:6s}   [{tag}]")

    # 4. error buckets ---------------------------------------------------
    buckets = error_buckets(true_idx, pred_idx, conf, test_df,
                            hi_conf=args.hi_conf, lo_conf=args.lo_conf)
    print("\n" + "=" * 78)
    print("ERROR BUCKETS")
    print("=" * 78)
    print(f"  E1  wrong sub, right top (cheap)   : {len(buckets['E1'])}")
    print(f"  E2  wrong top-level (expensive)    : {len(buckets['E2'])}")
    print(f"  E3  high-confidence wrong (>= {args.hi_conf}) : {len(buckets['E3'])}")
    print(f"  E4  low-confidence correct (<= {args.lo_conf}): {len(buckets['E4'])}")
    for name, recs in buckets.items():
        dfb = pd.DataFrame(recs, columns=["sound_id", "true", "pred", "conf"])
        dfb.to_csv(out / f"{name}.csv", index=False)

    print(f"\n  --- E3 (confidently wrong) top {args.top_k_examples}: the most"
          f" worrying errors ---")
    e3 = sorted(buckets["E3"], key=lambda r: r[3], reverse=True)[:args.top_k_examples]
    for sid, t, p, c in e3:
        tag = "CROSS-TOP" if BST[t] != BST[p] else "within-top"
        print(f"    {sid}  true={t:6s} pred={p:6s} conf={c:.3f}  [{tag}]")

    # 5. headline read ---------------------------------------------------
    attr = hf_shortfall_attribution(true_idx, pred_idx)
    print("\n" + "=" * 78)
    print("HEADLINE READ — is the remaining loss fixable or intrinsic?")
    print("=" * 78)
    print(f"  test clips                  : {attr['n']}")
    print(f"  correct                     : {attr['n_correct']}  "
          f"({attr['n_correct']/attr['n']:.1%})")
    print(f"  E1 (wrong sub, right top)   : {attr['n_E1']}")
    print(f"  E2 (wrong top)              : {attr['n_E2']}")
    print(f"  credit lost to E1           : {attr['credit_lost_E1']:.1f}")
    print(f"  credit lost to E2           : {attr['credit_lost_E2']:.1f}")
    print(f"  -> {attr['frac_loss_from_E1']:.0%} of the shortfall is WITHIN-family (E1)")
    print(f"  -> {attr['frac_loss_from_E2']:.0%} of the shortfall is CROSS-family (E2)")
    print()
    if attr["frac_loss_from_E1"] >= 0.5:
        print("  INTERPRETATION: most loss is within-family (right top, wrong sub).")
        print("  This is the more FIXABLE regime — class-bias / calibration / per-")
        print("  class threshold tuning on OOF preds can buy some of it back, since")
        print("  the model already lands in the right family.")
    else:
        print("  INTERPRETATION: a large share of loss is CROSS-family (wrong top).")
        print("  These are full-penalty errors and are closer to INTRINSIC. Calibration")
        print("  won't fix wrong-family predictions; look instead at whether specific")
        print("  top-level pairs dominate the off-diagonal above, and whether the audio")
        print("  vs text pathway disagrees on those (Priority 6 territory).")
    print("\nAll tables saved to:", out.resolve())


if __name__ == "__main__":
    main()
