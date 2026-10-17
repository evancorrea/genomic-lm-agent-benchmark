"""Run frozen, local NT-v2-50M inference; cache vectors outside agent-visible tools."""
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
PARENT = ROOT.parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    model_path = PARENT / ".cache" / "model"
    assert model_path.joinpath("model.safetensors").exists(), "Run parent scripts/prepare_model.py first"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True,
                                                 local_files_only=True, use_safetensors=True).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    start = time.monotonic()
    target = ROOT / ".cache"
    target.mkdir(exist_ok=True)
    shapes = {}
    for task in ("promoter", "enhancer", "open_chromatin"):
        for split in ("train", "test"):
            frame = pd.read_csv(ROOT / "data" / f"{task}_{split}.csv")
            vectors = []
            for i in range(0, len(frame), 8):
                tokens = tokenizer(frame.seq.iloc[i:i + 8].tolist(), return_tensors="pt",
                                   padding=True, truncation=True, max_length=tokenizer.model_max_length).to(device)
                with torch.inference_mode():
                    hidden = model(**tokens, output_hidden_states=True).hidden_states[-1]
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vectors.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
            features = np.concatenate(vectors).astype(np.float32)
            np.save(target / f"{task}_{split}.npy", features)
            shapes[f"{task}_{split}"] = list(features.shape)
            print(task, split, shapes[f"{task}_{split}"], flush=True)
    provenance = {"model_revision": json.loads((PARENT / "model_provenance.json").read_text())["revision"],
                  "dataset_revisions": {task: info["revision"] for task, info in
                                        json.loads((ROOT / "data" / "provenance.json").read_text())["datasets"].items()},
                  "pooling": "attention-mask-weighted last-hidden mean, special tokens included",
                  "device": device, "batch_size": 8, "torch": torch.__version__,
                  "inference_seconds_excluding_model_load": round(time.monotonic() - start, 2),
                  "shapes": shapes,
                  "sha256": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.npy")}}
    (ROOT / "embedding_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Inference seconds:", provenance["inference_seconds_excluding_model_load"])


if __name__ == "__main__":
    main()
