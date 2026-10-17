"""Launch one-prompt, independent Pi sessions for each task/arm/trial."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("promoter", "enhancer", "open_chromatin")
PROMPT = (ROOT / "task_prompt.txt").read_text()
MODEL = "openai/gpt-6-sol"
TOOLS = "describe_task,validate_feature,submit_pipeline"


def run(task, arm, trial):
    run_id = f"{task}-{arm}-{trial:02d}"
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "sessions").mkdir()
    command = ["pi", "--mode", "json", "--model", MODEL, "--thinking", "medium",
               "--no-extensions", "--extension", str(ROOT / "extensions" / "benchmark.ts"),
               "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-context-files",
               "--no-themes", "--no-approve", "--tools", TOOLS,
               "--session-dir", str(directory / "sessions"), "--session-id", run_id,
               "--name", run_id, PROMPT]
    env = dict(os.environ, BENCH_ROOT=str(ROOT), BENCH_TASK=task, BENCH_ARM=arm)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    status, code = "finished", None
    with (directory / "events.jsonl").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                  stderr=stderr, timeout=600).returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    events = []
    for line in (directory / "events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    manifest = {"run_id": run_id, "task": task, "arm": arm, "trial": trial,
                "paired_run_id": f"{task}-{'glm' if arm == 'control' else 'control'}-{trial:02d}",
                "parent_run_id": None, "agent_model": MODEL, "thinking": "medium",
                "dataset_revision": json.loads((ROOT / "data/provenance.json").read_text())["datasets"][task]["revision"],
                "genomic_model_revision": json.loads((ROOT / "embedding_provenance.json").read_text())["model_revision"] if arm == "glm" else None,
                "started_at": started, "duration_seconds": round(time.monotonic() - start, 3),
                "status": status, "exit_code": code,
                "agent_settled": any(e.get("type") == "agent_settled" for e in events),
                "session_id": next((e.get("id") for e in events if e.get("type") == "session"), None),
                "session_files": [str(p.relative_to(ROOT)) for p in (directory / "sessions").rglob("*.jsonl")],
                "event_file": str((directory / "events.jsonl").relative_to(ROOT)),
                "tool_sequence": [e["toolName"] for e in events if e.get("type") == "tool_execution_start"],
                "extension_errors": [e.get("error") for e in events if e.get("type") == "extension_error"]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(run_id, "exit", code, "settled", manifest["agent_settled"],
          "tools", manifest["tool_sequence"], flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3, help="Total repetitions per task/arm")
    args = parser.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    jobs = [(task, arm, trial) for task in TASKS for trial in range(1, args.trials + 1)
            for arm in ("control", "glm")
            if not (ROOT / "runs" / f"{task}-{arm}-{trial:02d}").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: run(*job), jobs))
    manifests = [json.loads(p.read_text()) for p in sorted((ROOT / "runs").glob("*/manifest.json"))]
    (ROOT / "runs" / "index.json").write_text(json.dumps(manifests, indent=2) + "\n")
