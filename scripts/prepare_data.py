"""Download a pinned public promoter dataset and select a small balanced pilot subset."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET = "katarinagresova/Genomic_Benchmarks_human_nontata_promoters"
API = f"https://huggingface.co/api/datasets/{DATASET}"
REVISION = "e0003669df0a180c8570ebe091ed91f20fa080ec"
SEED = 20261017
PER_CLASS = {"train": 64, "test": 32}


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with path.with_suffix(path.suffix + ".part").open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    path.with_suffix(path.suffix + ".part").replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    revision = REVISION
    manifest_path = ROOT / "data" / "provenance.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    response = requests.get(f"{API}/tree/{revision}/data", timeout=30)
    response.raise_for_status()
    tree = response.json()
    manifest = {"dataset": DATASET, "revision": revision, "seed": SEED,
                "per_class": PER_CLASS, "source_files": {}, "subset_sha256": {}}
    for split in ("train", "test"):
        path_in_repo = next(x["path"] for x in tree if x["path"].startswith(f"data/{split}-") and x["path"].endswith(".parquet"))
        source = ROOT / "data" / "source" / Path(path_in_repo).name
        download(f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path_in_repo}", source)
        source_hash = digest(source)
        expected_hash = previous.get("source_files", {}).get(split, {}).get("sha256")
        if expected_hash and source_hash != expected_hash:
            raise ValueError(f"Source hash changed: {source}")
        frame = pd.read_parquet(source)
        selected = []
        for label in (0, 1):
            subset = frame.loc[frame.label == label].sample(n=PER_CLASS[split], random_state=SEED + label)
            selected.append(subset)
        sample = pd.concat(selected).sample(frac=1, random_state=SEED).reset_index(drop=True)
        sample["seq"] = sample["seq"].str.upper()
        sample.insert(0, "id", [f"{split}-{i:03d}" for i in range(len(sample))])
        subset = ROOT / "data" / f"{split}.csv"
        sample.to_csv(subset, index=False)
        subset_hash = digest(subset)
        if previous.get("subset_sha256", {}).get(split) not in (None, subset_hash):
            raise ValueError(f"Subset hash changed: {subset}")
        manifest["subset_sha256"][split] = subset_hash
        manifest["source_files"][split] = {"path": str(source.relative_to(ROOT)), "sha256": source_hash,
                                           "rows": len(frame), "selected_rows": len(sample),
                                           "sequence_lengths": sorted(sample.seq.str.len().unique().tolist())}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
