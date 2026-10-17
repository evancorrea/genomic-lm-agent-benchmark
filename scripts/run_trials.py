"""Run paired, one-prompt Pi trials; preserve session, JSON event trace, and provenance."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "task_prompt.txt").read_text()
TASK_ID = "public-promoter-nt50m-v1"
MODEL = "openai/gpt-6-sol"
TOOLS = {"control": ["describe_dataset", "evaluate_gc_3mer"],
         "glm": ["describe_dataset", "evaluate_gc_3mer", "evaluate_nt_embeddings"]}


def run(arm, trial):
    run_id = f"{arm}-trial-{trial:02d}"
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "sessions").mkdir()
    command = ["pi", "--mode", "json", "--model", MODEL, "--thinking", "medium",
               "--no-extensions", "--extension", str(ROOT / "extensions" / "benchmark.ts"),
               "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-context-files",
               "--no-themes", "--no-approve", "--tools", ",".join(TOOLS[arm]),
               "--session-dir", str(directory / "sessions"), "--session-id", run_id,
               "--name", run_id, PROMPT]
    env = dict(os.environ, BENCH_ROOT=str(ROOT), BENCH_ARM=arm)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    code = None
    status = "finished"
    with (directory / "events.jsonl").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                  stderr=stderr, timeout=600).returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    records = []
    for line in (directory / "events.jsonl").read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    manifest = {"run_id": run_id, "task_id": TASK_ID, "arm": arm,
                "trial": trial, "parent_run_id": None, "paired_run_id": f"{'glm' if arm == 'control' else 'control'}-trial-{trial:02d}",
                "model": MODEL, "thinking": "medium", "tools": TOOLS[arm],
                "dataset_revision": json.loads((ROOT / "data/provenance.json").read_text())["revision"],
                "genomic_model_revision": json.loads((ROOT / "model_provenance.json").read_text())["revision"] if arm == "glm" else None,
                "started_at": started, "duration_seconds": round(time.monotonic() - start, 3),
                "status": status, "exit_code": code, "agent_settled": any(x.get("type") == "agent_settled" for x in records),
                "session_id": next((x.get("id") for x in records if x.get("type") == "session"), None),
                "session_files": [str(p.relative_to(ROOT)) for p in (directory / "sessions").rglob("*.jsonl")],
                "event_file": str((directory / "events.jsonl").relative_to(ROOT)),
                "event_count": len(records),
                "tool_sequence": [x.get("toolName") for x in records if x.get("type") == "tool_execution_start"],
                "extension_errors": [x.get("error") for x in records if x.get("type") == "extension_error"]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{run_id}: rc={code}, settled={manifest['agent_settled']}, tools={manifest['tool_sequence']}, {manifest['duration_seconds']}s", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Total paired trial IDs to create, not additional trials")
    args = parser.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    jobs = [(arm, trial) for trial in range(1, args.trials + 1) for arm in ("control", "glm")
            if not (ROOT / "runs" / f"{arm}-trial-{trial:02d}").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda job: run(*job), jobs))
    manifests = [json.loads(p.read_text()) for p in sorted((ROOT / "runs").glob("*/manifest.json"))]
    (ROOT / "runs" / "index.json").write_text(json.dumps(manifests, indent=2) + "\n")
