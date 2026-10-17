"""Fixed, auditable classifier protocol; run once before any Pi trial."""
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
K = 3
KMERS = ["".join(x) for x in product("ACGT", repeat=K)]
REGULARIZATION = [0.01, 0.1, 1.0, 10.0]
SEED = 421


def composition(sequences):
    features = []
    for seq in sequences:
        seq = str(seq).upper()
        counts = Counter(seq[i:i + K] for i in range(len(seq) - K + 1))
        features.append([(seq.count("G") + seq.count("C")) / len(seq)] +
                        [counts[kmer] / max(1, len(seq) - K + 1) for kmer in KMERS])
    return np.asarray(features, dtype=np.float64)


def pipeline(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=SEED))


def evaluate(name, train_x, test_x, train_y, test_y):
    fit_index, val_index = train_test_split(np.arange(len(train_y)), test_size=0.25,
                                            stratify=train_y, random_state=SEED)
    search = []
    for c in REGULARIZATION:
        model = pipeline(c).fit(train_x[fit_index], train_y[fit_index])
        probability = model.predict_proba(train_x[val_index])[:, 1]
        search.append({"C": c, "validation_auc": float(roc_auc_score(train_y[val_index], probability)),
                       "validation_accuracy": float(accuracy_score(train_y[val_index], probability >= 0.5))})
    # Select using training-only validation; deterministic tie-breaking favors stronger regularization.
    chosen = max(search, key=lambda x: (x["validation_auc"], x["validation_accuracy"], -x["C"]))
    model = pipeline(chosen["C"]).fit(train_x, train_y)
    probability = model.predict_proba(test_x)[:, 1]
    return {"features": name, "selected_C": chosen["C"],
            "validation_auc": chosen["validation_auc"],
            "validation_accuracy": chosen["validation_accuracy"],
            "test_auc": float(roc_auc_score(test_y, probability)),
            "test_accuracy": float(accuracy_score(test_y, probability >= 0.5)),
            "hyperparameter_search": search,
            "train_rows": len(train_y), "test_rows": len(test_y)}


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    assert len(set(train.seq) & set(test.seq)) == 0, "Exact duplicate sequences cross train/test"
    train_y = train.label.to_numpy()
    test_y = test.label.to_numpy()
    baseline = evaluate("gc_3mer", composition(train.seq), composition(test.seq), train_y, test_y)
    embeddings = np.load(ROOT / ".cache" / "embeddings.npz")
    glm = evaluate("nt_v2_50m", embeddings["train"], embeddings["test"], train_y, test_y)
    output = {"protocol": {"split": "official train/test, balanced deterministic subset; 25% of train for C selection",
                           "selection": "highest validation AUC, then validation accuracy, then smallest C",
                           "training": "standardize features and fit logistic regression; refit on full train",
                           "test": "evaluate exactly once after hyperparameter selection",
                           "seed": SEED, "kmer_length": K, "C_grid": REGULARIZATION},
              "gc_3mer": baseline, "nt_v2_50m": glm}
    target = ROOT / ".cache" / "evaluation.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
