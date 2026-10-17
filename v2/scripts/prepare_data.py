"""Download pinned public data and create balanced, disjoint v2 subsets."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "promoter": ("human_nontata_promoters", "e0003669df0a180c8570ebe091ed91f20fa080ec"),
    "enhancer": ("human_enhancers_cohn", "49ef8d0d80087ece7441096afcef0123b0781ce0"),
    "open_chromatin": ("human_ocr_ensembl", "3c07e4be6425be11073baa2daacf6c143c64e4ff"),
}
SEED = 20261017
PER_CLASS = {"train": 128, "test": 64}
SIMILARITY_K = 7
MAX_TEST_JACCARD = 0.8


def kmers(sequence):
    return {sequence[i:i + SIMILARITY_K] for i in range(len(sequence) - SIMILARITY_K + 1)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    tmp.replace(path)


def main():
    prior_path = ROOT / "data" / "provenance.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    manifest = {"seed": SEED, "per_class": PER_CLASS,
                "test_similarity_screen": {"k": SIMILARITY_K, "max_jaccard_exclusive": MAX_TEST_JACCARD,
                                           "includes_reverse_complement": True}, "datasets": {}}
    for task, (short_name, revision) in DATASETS.items():
        dataset = f"katarinagresova/Genomic_Benchmarks_{short_name}"
        url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/data"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        tree = response.json()
        result = {"repository": dataset, "revision": revision, "files": {}, "subsets": {}}
        for split in ("train", "test"):
            repo_path = next(item["path"] for item in tree if item["path"].startswith(f"data/{split}-")
                             and item["path"].endswith(".parquet"))
            source = ROOT / "data" / "source" / task / Path(repo_path).name
            download(f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{repo_path}", source)
            checksum = sha256(source)
            expected = prior.get("datasets", {}).get(task, {}).get("files", {}).get(split, {}).get("sha256")
            if expected and expected != checksum:
                raise ValueError(f"Upstream content changed: {source}")
            frame = pd.read_parquet(source)
            parts = []
            train_kmers = []
            if split == "test":
                train_frame = pd.read_csv(ROOT / "data" / f"{task}_train.csv")
                reverse = str.maketrans("ACGT", "TGCA")
                train_kmers = [kmers(seq) for seq in train_frame.seq]
            for label in (0, 1):
                if split == "train":
                    parts.append(frame.loc[frame.label == label].sample(n=PER_CLASS[split],
                                                                         random_state=SEED + label))
                    continue
                candidates = frame.loc[frame.label == label].sample(frac=1, random_state=SEED + label)
                accepted = []
                for index, row in candidates.iterrows():
                    seq = str(row.seq).upper()
                    alternatives = (kmers(seq), kmers(seq.translate(reverse)[::-1]))
                    maximum = max(len(a & b) / len(a | b) for a in alternatives for b in train_kmers)
                    if maximum < MAX_TEST_JACCARD:
                        accepted.append(index)
                    if len(accepted) == PER_CLASS[split]:
                        break
                if len(accepted) != PER_CLASS[split]:
                    raise ValueError(f"Not enough low-similarity test sequences for {task} label {label}")
                parts.append(frame.loc[accepted])
            subset = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
            subset["seq"] = subset.seq.str.upper()
            subset.insert(0, "id", [f"{task}-{split}-{i:04d}" for i in range(len(subset))])
            destination = ROOT / "data" / f"{task}_{split}.csv"
            subset.to_csv(destination, index=False)
            subset_hash = sha256(destination)
            old = prior.get("datasets", {}).get(task, {}).get("subsets", {}).get(split, {}).get("sha256")
            if old and old != subset_hash:
                raise ValueError(f"Local subset changed: {destination}")
            result["files"][split] = {"path": str(source.relative_to(ROOT)), "sha256": checksum,
                                      "source_rows": len(frame)}
            result["subsets"][split] = {"path": str(destination.relative_to(ROOT)), "sha256": subset_hash,
                                        "rows": len(subset), "length_min": int(subset.seq.str.len().min()),
                                        "length_max": int(subset.seq.str.len().max())}
        train = pd.read_csv(ROOT / "data" / f"{task}_train.csv")
        test = pd.read_csv(ROOT / "data" / f"{task}_test.csv")
        reverse_complement = str.maketrans("ACGT", "TGCA")
        train_sequences = set(train.seq) | {s.translate(reverse_complement)[::-1] for s in train.seq}
        overlap = set(test.seq) & train_sequences
        if overlap:
            raise ValueError(f"Exact/reverse-complement leakage for {task}: {len(overlap)}")
        result["exact_or_reverse_complement_overlap"] = 0
        manifest["datasets"][task] = result
        print(task, "train", len(train), "test", len(test), "lengths", result["subsets"]["train"]["length_min"],
              result["subsets"]["train"]["length_max"], flush=True)
    prior_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
