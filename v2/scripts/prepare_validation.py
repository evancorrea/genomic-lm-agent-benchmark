"""Precompute only TRAINING-VALIDATION metrics; never touch the held-out test."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from analysis import ROOT, TASKS, FEATURES, CS, THRESHOLDS, SEED, table, features, fit, metrics


def main():
    output = {"protocol": {"seed": SEED, "train_validation_fraction": 0.25,
                           "C_values": CS, "thresholds": THRESHOLDS,
                           "validation_metric": "average precision (AUPRC)",
                           "feature_choice": "highest validation AUPRC; tie: highest validation F1, then simpler family",
                           "threshold_choice": "highest validation F1 among allowed thresholds for chosen C"},
              "tasks": {}}
    for task in TASKS:
        labels = table(task, "train").label.to_numpy()
        train_i, validation_i = train_test_split(np.arange(len(labels)), test_size=0.25,
                                                 stratify=labels, random_state=SEED)
        output["tasks"][task] = {}
        for feature in FEATURES:
            x = features(task, "train", feature)
            configurations = []
            for c in CS:
                model = fit(x[train_i], labels[train_i], c)
                probability = model.predict_proba(x[validation_i])[:, 1]
                for threshold in THRESHOLDS:
                    configurations.append({"C": c, "threshold": threshold,
                                           **metrics(labels[validation_i], probability, threshold)})
            output["tasks"][task][feature] = configurations
            print(task, feature, "best validation AUPRC", max(c["auprc"] for c in configurations), flush=True)
    target = ROOT / ".cache" / "validation.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
