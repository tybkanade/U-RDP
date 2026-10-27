# -*- coding: utf-8 -*-
"""Accuracy 与宏平均 Precision / Recall / F1 / AUROC（one-vs-rest）。"""
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

KEYS = ("acc", "precision", "recall", "f1", "auroc")


def classification_metrics(y_true, probs, num_classes: int) -> Dict:
    y_true, probs = np.asarray(y_true), np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(1)
    labels = np.arange(num_classes)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    try:
        auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        auroc = float("nan")
    return {"acc": float((y_pred == y_true).mean()), "precision": float(p), "recall": float(r),
            "f1": float(f1), "auroc": auroc,
            "cm": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def fmt(m: Dict) -> str:
    return " ".join(f"{k}={m[k]:.4f}" for k in KEYS)


def format_cm(cm, names: Sequence[str]) -> str:
    w = max(8, max(len(n) for n in names) + 1)
    lines = [" " * w + "".join(f"{n:>{w}}" for n in names)]
    lines += [f"{n:>{w}}" + "".join(f"{int(v):>{w}}" for v in row) for n, row in zip(names, cm)]
    return "\n".join(lines)


def mean_std(results: List[Dict]) -> Dict:
    return {"mean": {k: float(np.mean([r[k] for r in results])) for k in KEYS},
            "std": {k: float(np.std([r[k] for r in results])) for k in KEYS}}


def print_mean_std(title: str, summary: Dict):
    print(f"\n===== {title} =====")
    for k in KEYS:
        print(f"{k:>9}: {100 * summary['mean'][k]:.2f} ± {100 * summary['std'][k]:.2f}")
