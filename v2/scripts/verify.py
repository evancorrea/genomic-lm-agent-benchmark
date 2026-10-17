"""Verify pinned inputs, no similar sampled test sequences, and saved run integrity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_file(path, expected):
    assert path.exists(), f"Missing {path}"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, f"SHA-256 mismatch: {path}"


def main():
    data = json.loads((ROOT / "data/provenance.json").read_text())
    for task, meta in data["datasets"].items():
        for split in ("train", "test"):
            check_file(ROOT / meta["files"][split]["path"], meta["files"][split]["sha256"])
            check_file(ROOT / meta["subsets"][split]["path"], meta["subsets"][split]["sha256"])
    embedding = json.loads((ROOT / "embedding_provenance.json").read_text())
    for name, checksum in embedding["sha256"].items():
        check_file(ROOT / ".cache" / f"{name}.npy", checksum)
    audit = json.loads((ROOT / "similarity_audit.json").read_text())
    assert all(info["test_rows_nearest_at_least_0.8"] == 0 for info in audit["tasks"].values())
    results = json.loads((ROOT / "results.json").read_text())
    assert len(results) == 18
    for run in results:
        assert run["exit_code"] == 0 and run["agent_settled"] and not run["extension_errors"]
        assert run["score"] == run["possible"] == 9 and all(run["audit"].values())
        assert run["tool_errors"] == 0 and len(run["session_files"]) == 1
        assert (ROOT / run["event_file"]).exists() and (ROOT / run["session_files"][0]).exists()
        assert run["tool_sequence"].count("submit_pipeline") == 1
        assert run["tool_sequence"].count("validate_feature") == (3 if run["arm"] == "glm" else 2)
    print("Verified input/embedding checksums, similarity screen, and 18 completed Pi traces/sessions.")


if __name__ == "__main__":
    main()
