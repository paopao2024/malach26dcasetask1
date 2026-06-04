"""
DCASE 2026 Task 1 -- official hierarchical metric.

Verbatim copy of hierarchical_prf_weighted and the surrounding macro loop from
the organizers' evaluate.py (compute_metrics). Helpers (get_top_level,
extend_subcat, intersection) are their utils.py helpers, reconstructed exactly
from how they are used.

Every script that monitors or reports the leaderboard metric should import
macro_hPRF from this file so the metric is identical across the project.

Verification: running this code against the published reference table in the
task description matches every deterministic row exactly (Perfect 1.000,
All-top-correct 0.375, All-both-wrong 0.000, Half-top-correct 0.688) and
matches the random rows within sampling noise.
"""

import numpy as np


# ---- helpers (verbatim behaviour of the organizers' utils.py) ----
def get_top_level(label):
    return label.split('-')[0]


def extend_subcat(label):
    return [label, label.split('-')[0]]


def intersection(a, b):
    return list(set(a) & set(b))


# ---- verbatim from evaluate.py ----
def hierarchical_prf_weighted(subcat, predictions_gt, lambda_param=0.75):
    hPP = []
    hRR = []
    for count, (prediction, gt) in enumerate(predictions_gt):
        pi = extend_subcat(prediction)
        ti = extend_subcat(gt)
        pi_intersection_ti = intersection(pi, ti)

        if subcat == prediction:
            w = 1 if prediction == gt else (lambda_param if get_top_level(prediction) == get_top_level(gt) else 0)
            hP = (w * len(pi_intersection_ti)) / len(pi)
            hPP.append(hP)

        if subcat == gt:
            w = 1 if prediction == gt else (lambda_param if get_top_level(prediction) == get_top_level(gt) else 0)
            hR = (w * len(pi_intersection_ti)) / len(ti)
            hRR.append(hR)

    classP = sum(hPP) / len(hPP)
    classR = sum(hRR) / len(hRR)
    if classR == 0 and classP == 0:
        classF = 0
    else:
        classF = 2 * classP * classR / (classP + classR)
    return classP, classR, classF


# ---- macro structure mirroring compute_metrics in evaluate.py ----
def macro_hPRF(true_labels, pred_labels, lambda_param=0.75):
    """
    true_labels, pred_labels : lists of class-name strings (e.g. ["m-sp", "fx-a", ...]).
    Returns (hP, hR, hF) as floats. The leaderboard metric is hF.
    """
    pred_gt_pairs = list(zip(pred_labels, true_labels))    # tuples are (pred, gt)
    classes = list(set(true_labels))
    hPs, hRs, hFs = [], [], []
    for c in classes:
        try:
            p, r, f = hierarchical_prf_weighted(c, pred_gt_pairs, lambda_param=lambda_param)
            if not (np.isnan(p) or np.isnan(r) or np.isnan(f)):
                hPs.append(p); hRs.append(r); hFs.append(f)
        except Exception:
            continue
    hP = float(np.mean(hPs)) if hPs else 0.0
    hR = float(np.mean(hRs)) if hRs else 0.0
    hF = float(np.mean(hFs)) if hFs else 0.0
    return hP, hR, hF
