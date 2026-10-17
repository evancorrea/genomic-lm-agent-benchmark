"""One fixed-pipeline held-out evaluation invoked only by the extension's submit tool."""
import json
import sys

from analysis import TASKS, FEATURES, CS, THRESHOLDS, table, features, fit, metrics


def main():
    task, feature, c_arg, threshold_arg = sys.argv[1:]
    c, threshold = float(c_arg), float(threshold_arg)
    if task not in TASKS or feature not in FEATURES or c not in CS or threshold not in THRESHOLDS:
        raise ValueError("Invalid submission")
    train_y, test_y = table(task, "train").label.to_numpy(), table(task, "test").label.to_numpy()
    model = fit(features(task, "train", feature), train_y, c)
    probability = model.predict_proba(features(task, "test", feature))[:, 1]
    print(json.dumps({"task": task, "features": feature, "C": c, "threshold": threshold,
                      "held_out": metrics(test_y, probability, threshold),
                      "test_rows": len(test_y)}))


if __name__ == "__main__":
    main()
