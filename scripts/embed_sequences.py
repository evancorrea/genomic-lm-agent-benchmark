"""Precompute mean-pooled NT-v2-50M embeddings for the pinned pilot subset."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    model_dir = ROOT / ".cache" / "model"
    assert model_dir.joinpath("model.safetensors").exists(), "Run prepare_model.py first"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), trust_remote_code=True,
                                                 local_files_only=True, use_safetensors=True).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    started = time.monotonic()
    result = {}
    for split in ("train", "test"):
        data = pd.read_csv(ROOT / "data" / f"{split}.csv")
        vectors = []
        for start in range(0, len(data), 8):
            sequences = data.seq.iloc[start:start + 8].tolist()
            tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True,
                               max_length=tokenizer.model_max_length).to(device)
            with torch.inference_mode():
                hidden = model(**tokens, output_hidden_states=True).hidden_states[-1]
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.append(pooled.cpu().numpy())
            print(f"{split}: {min(start + 8, len(data))}/{len(data)}", flush=True)
        result[split] = np.concatenate(vectors).astype(np.float32)
    target = ROOT / ".cache" / "embeddings.npz"
    np.savez_compressed(target, **result)
    provenance = {
        "model_revision": json.loads((ROOT / "model_provenance.json").read_text())["revision"],
        "data_revision": json.loads((ROOT / "data" / "provenance.json").read_text())["revision"],
        "pooling": "attention-mask-weighted mean of last hidden layer, including special tokens",
        "device": device, "batch_size": 8, "torch": torch.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "shape": {split: list(array.shape) for split, array in result.items()},
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (ROOT / "embedding_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
