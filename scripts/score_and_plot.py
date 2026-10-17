"""Grade answer fidelity and plot paired agent runs; never treat repeats as data resamples."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ("baseline_test_auc", "selected_test_auc", "delta_test_auc",
           "baseline_test_accuracy", "selected_test_accuracy", "delta_test_accuracy")


def score(directory, evaluation):
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    messages = [e["message"] for e in events if e.get("type") == "message_end"
                and e.get("message", {}).get("role") == "assistant"]
    text = "".join(piece.get("text", "") for piece in messages[-1].get("content", [])
                   if piece.get("type") == "text") if messages else ""
    try:
        answer = json.loads(text)
        if not isinstance(answer, dict):
            raise ValueError("not an object")
        error = None
    except (ValueError, json.JSONDecodeError) as exc:
        answer, error = {}, str(exc)
    baseline = evaluation["gc_3mer"]
    expected_choice = "gc_3mer" if manifest["arm"] == "control" else (
        "nt_v2_50m" if evaluation["nt_v2_50m"]["validation_auc"] > baseline["validation_auc"] else "gc_3mer")
    selected = evaluation[expected_choice]
    expected = {"baseline_test_auc": baseline["test_auc"], "selected_test_auc": selected["test_auc"],
                "delta_test_auc": selected["test_auc"] - baseline["test_auc"],
                "baseline_test_accuracy": baseline["test_accuracy"],
                "selected_test_accuracy": selected["test_accuracy"],
                "delta_test_accuracy": selected["test_accuracy"] - baseline["test_accuracy"]}
    checks = {"selected_features": answer.get("selected_features") == expected_choice}
    for key in NUMERIC:
        value = answer.get(key)
        checks[key] = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
                      and math.isclose(value, expected[key], rel_tol=1e-4, abs_tol=1e-8)
    usage = []
    for filename in manifest["session_files"]:
        for line in (ROOT / filename).read_text().splitlines():
            entry = json.loads(line)
            if entry.get("type") == "message" and entry["message"].get("role") == "assistant":
                usage.append(entry["message"].get("usage", {}))
    return {**manifest, "score": sum(checks.values()), "possible": len(checks),
            "field_correct": checks, "parse_error": error,
            "final_stop_reason": messages[-1].get("stopReason") if messages else None,
            "selected_features": answer.get("selected_features"),
            "selected_test_auc": answer.get("selected_test_auc"),
            "selected_test_accuracy": answer.get("selected_test_accuracy"),
            "answer": answer, "assistant_turns": len(messages),
            "tool_calls": sum(e.get("type") == "tool_execution_start" for e in events),
            "retry_count": sum(e.get("type") == "auto_retry_start" for e in events),
            "input_tokens": sum(x.get("input", 0) for x in usage),
            "output_tokens": sum(x.get("output", 0) for x in usage),
            "reported_pi_cost_usd": sum(x.get("cost", {}).get("total", 0) for x in usage)}


def main():
    evaluation = json.loads((ROOT / ".cache" / "evaluation.json").read_text())
    results = [score(directory, evaluation) for directory in sorted((ROOT / "runs").iterdir()) if directory.is_dir()]
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    columns = ["run_id", "arm", "trial", "score", "possible", "selected_features", "selected_test_auc",
               "selected_test_accuracy", "duration_seconds", "tool_calls", "retry_count",
               "reported_pi_cost_usd", "exit_code", "agent_settled"]
    with (ROOT / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    metrics = [("selected_test_auc", "Held-out AUROC"),
               ("selected_test_accuracy", "Held-out accuracy (threshold 0.5)"),
               ("score", "Correct answer fields / 7"),
               ("reported_pi_cost_usd", "Reported Pi model cost (USD)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), constrained_layout=True)
    for ax, (field, label) in zip(axes, metrics):
        for i, (arm, color) in enumerate((("control", "#68899c"), ("glm", "#7554a3"))):
            subset = sorted((r for r in results if r["arm"] == arm), key=lambda r: r["trial"])
            values = [r[field] if isinstance(r[field], (int, float)) else float("nan") for r in subset]
            ax.scatter(np.linspace(i - .12, i + .12, len(values)), values, color=color, s=55, zorder=3)
            ax.plot([i - .2, i + .2], [np.nanmean(values)] * 2, color="black", linewidth=2)
        ax.set_xticks([0, 1], ["No GLM tool", "NT-v2 tool"])
        ax.set_ylabel(label)
        ax.set_xlim(-.4, 1.4)
        ax.grid(axis="y", alpha=.2)
        if field == "score": ax.set_ylim(0, 7.6)
        elif field in ("selected_test_auc", "selected_test_accuracy"): ax.set_ylim(.7, 1.0)
        else: ax.set_ylim(0, max(r[field] for r in results) * 1.25)
    fig.suptitle("One fixed promoter subset · five agent repetitions per arm · GPT-6 Sol, medium thinking")
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)
    for arm in ("control", "glm"):
        subset = [r for r in results if r["arm"] == arm]
        print(arm, "scores", [r["score"] for r in subset],
              "AUROC", [r["selected_test_auc"] for r in subset],
              "accuracy", [r["selected_test_accuracy"] for r in subset])


if __name__ == "__main__":
    main()
