"""Flag unusually similar sampled train/test sequences using an approximate 7-mer screen."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("promoter", "enhancer", "open_chromatin")
K = 7


def kmers(sequence):
    return {sequence[i:i + K] for i in range(len(sequence) - K + 1)}


def main():
    findings = {"k": K, "method": "maximum Jaccard of 7-mer sets per test sequence against sampled train sequences; also reverse complements",
                "warning": "This is only a heuristic screen, not a coordinate-aware or homology-aware split.", "tasks": {}}
    reverse = str.maketrans("ACGT", "TGCA")
    for task in TASKS:
        train = pd.read_csv(ROOT / "data" / f"{task}_train.csv")
        test = pd.read_csv(ROOT / "data" / f"{task}_test.csv")
        train_sets = [kmers(seq) for seq in train.seq]
        test_best = []
        for seq in test.seq:
            variants = (kmers(seq), kmers(seq.translate(reverse)[::-1]))
            test_best.append(max(len(a & b) / len(a | b) for a in variants for b in train_sets))
        findings["tasks"][task] = {"maximum": max(test_best), "mean_nearest": sum(test_best) / len(test_best),
                                    "test_rows_nearest_at_least_0.8": sum(x >= .8 for x in test_best),
                                    "test_rows_nearest_at_least_0.9": sum(x >= .9 for x in test_best)}
        print(task, findings["tasks"][task], flush=True)
    (ROOT / "similarity_audit.json").write_text(json.dumps(findings, indent=2) + "\n")


if __name__ == "__main__":
    main()
