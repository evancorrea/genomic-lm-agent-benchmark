"""Shared, deterministic feature and classifier definitions for v2."""
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, accuracy_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("promoter", "enhancer", "open_chromatin")
FEATURES = ("gc_3mer", "gc_5mer", "nt_v2_50m")
CS = (0.01, 0.1, 1.0)
THRESHOLDS = (0.35, 0.5, 0.65)
SEED = 421


def table(task, split):
    if task not in TASKS or split not in ("train", "test"):
        raise ValueError("Invalid task or split")
    return pd.read_csv(ROOT / "data" / f"{task}_{split}.csv")


def features(task, split, name):
    if name not in FEATURES:
        raise ValueError("Invalid feature family")
    if name == "nt_v2_50m":
        return np.load(ROOT / ".cache" / f"{task}_{split}.npy")
    k = 3 if name == "gc_3mer" else 5
    kmers = ["".join(chars) for chars in product("ACGT", repeat=k)]
    vectors = []
    for seq in table(task, split).seq:
        seq = seq.upper()
        counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
        vectors.append([(seq.count("G") + seq.count("C")) / len(seq)] +
                       [counts[token] / max(1, len(seq) - k + 1) for token in kmers])
    return np.asarray(vectors, dtype=np.float32)


def fit(x, y, c):
    if c not in CS:
        raise ValueError("Invalid C")
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="liblinear",
                                                               max_iter=2000, random_state=SEED)).fit(x, y)


def metrics(y, probability, threshold):
    if threshold not in THRESHOLDS:
        raise ValueError("Invalid threshold")
    predicted = probability >= threshold
    return {"auprc": float(average_precision_score(y, probability)),
            "auroc": float(roc_auc_score(y, probability)),
            "f1": float(f1_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "accuracy": float(accuracy_score(y, predicted))}
