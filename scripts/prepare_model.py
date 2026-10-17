"""Pin and download only the files needed for NT-v2-50M inference."""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
FILES = ["config.json", "esm_config.py", "modeling_esm.py", "model.safetensors",
         "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]


def main():
    revision = REVISION
    manifest_path = ROOT / "model_provenance.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    target = ROOT / ".cache" / "model"
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        path = target / name
        if not path.exists():
            url = f"https://huggingface.co/{MODEL}/resolve/{revision}/{name}"
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                temp = path.with_suffix(path.suffix + ".part")
                with temp.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                temp.replace(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = previous.get("files", {}).get(name, {}).get("sha256")
        if expected and checksum != expected:
            raise ValueError(f"Model file checksum mismatch: {name}")
        hashes[name] = {"bytes": path.stat().st_size, "sha256": checksum}
        print(f"{name}: {path.stat().st_size / 1024**2:.1f} MiB", flush=True)
    manifest_path.write_text(json.dumps({"model": MODEL, "revision": revision,
                                          "files": hashes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
