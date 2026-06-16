import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _safe_auc(y_true, prob):
    try:
        return float(roc_auc_score(y_true, prob))
    except ValueError:
        return float("nan")


def binary_metrics(y_true, prob_pos, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    y_pred = (prob_pos >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auc": _safe_auc(y_true, prob_pos),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sen": float(sen),
        "spe": float(spe),
    }


def multiclass_metrics(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = prob.argmax(axis=1)
    try:
        auc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
