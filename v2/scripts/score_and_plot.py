"""Audit one-shot submissions and plot unique task/arm outcomes (not pseudoreplicates)."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER = ("gc_3mer", "gc_5mer", "nt_v2_50m")
METRICS = ("auprc", "auroc", "f1", "accuracy")


def close(a, b):
    return isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) \
           and math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-8)


def optimal_config(validation, task, arm):
    available = FEATURE_ORDER if arm == "glm" else FEATURE_ORDER[:2]
    entries = [(name, row) for name in available for row in validation["tasks"][task][name]]
    return max(entries, key=lambda x: (x[1]["auprc"], x[1]["f1"],
                                       -FEATURE_ORDER.index(x[0]), -x[1]["C"],
                                       -abs(x[1]["threshold"] - .5), -x[1]["threshold"]))


def grade(directory, validation):
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    calls = [e for e in events if e.get("type") == "tool_execution_start"]
    submissions = [e for e in events if e.get("type") == "tool_execution_end"
                   and e.get("toolName") == "submit_pipeline" and not e.get("isError")]
    submitted = json.loads(submissions[0]["result"]["content"][0]["text"]) if len(submissions) == 1 else None
    submitted_args = next((e["args"] for e in calls if e.get("toolName") == "submit_pipeline"), {})
    validation_row = next((row for row in validation["tasks"][manifest["task"]].get(
        submitted_args.get("features", ""), [])
        if row["C"] == submitted_args.get("C") and row["threshold"] == submitted_args.get("threshold")), None)
    expected_feature, expected_row = optimal_config(validation, manifest["task"], manifest["arm"])
    optimal = bool(validation_row and submitted_args.get("features") == expected_feature
                   and submitted_args["C"] == expected_row["C"]
                   and submitted_args["threshold"] == expected_row["threshold"])
    messages = [e["message"] for e in events if e.get("type") == "message_end"
                and e.get("message", {}).get("role") == "assistant"]
    text = "".join(block.get("text", "") for block in messages[-1]["content"]
                   if block.get("type") == "text") if messages else ""
    try:
        answer = json.loads(text)
        if not isinstance(answer, dict): raise ValueError("Not a JSON object")
        parse_error = None
    except ValueError as error:
        answer, parse_error = {}, str(error)
    checks = {"selected_features": answer.get("selected_features") == submitted_args.get("features")
              and submitted is not None,
              "C": close(answer.get("C"), submitted_args.get("C")) if submitted_args else False,
              "threshold": close(answer.get("threshold"), submitted_args.get("threshold")) if submitted_args else False,
              "validation_auprc": close(answer.get("validation_auprc"), validation_row["auprc"]) if validation_row else False,
              "validation_f1": close(answer.get("validation_f1"), validation_row["f1"]) if validation_row else False}
    for name in METRICS:
        checks["heldout_" + name] = close(answer.get("heldout_" + name), submitted["held_out"][name]) if submitted else False
    validated = [e["args"].get("features") for e in calls if e.get("toolName") == "validate_feature"]
    required = set(FEATURE_ORDER if manifest["arm"] == "glm" else FEATURE_ORDER[:2])
    audit = {"validated_all_available": set(validated) == required and len(validated) == len(required),
             "one_test_submission": sum(e.get("toolName") == "submit_pipeline" for e in calls) == 1,
             "correct_arm_exposure": not (manifest["arm"] == "control" and "nt_v2_50m" in validated),
             "optimal_validation_choice": optimal}
    usage = []
    for session in manifest["session_files"]:
        for line in (ROOT / session).read_text().splitlines():
            entry = json.loads(line)
            if entry.get("type") == "message" and entry["message"].get("role") == "assistant":
                usage.append(entry["message"].get("usage", {}))
    return {**manifest, "score": sum(checks.values()), "possible": len(checks),
            "field_correct": checks, "audit": audit, "parse_error": parse_error,
            "selected_features": submitted_args.get("features"),
            "C": submitted_args.get("C"), "threshold": submitted_args.get("threshold"),
            "held_out": submitted.get("held_out") if submitted else None,
            "final_answer": answer, "tool_calls": len(calls), "assistant_turns": len(messages),
            "tool_errors": sum(e.get("type") == "tool_execution_end" and e.get("isError") for e in events),
            "retry_count": sum(e.get("type") == "auto_retry_start" for e in events),
            "reported_pi_cost_usd": sum(u.get("cost", {}).get("total", 0) for u in usage)}


def plot(results):
    # Collapse repeated agent decisions to one outcome per task/arm: they share the same test set.
    unique = (pd.DataFrame([{"task": r["task"], "arm": r["arm"], **(r["held_out"] or {})}
                            for r in results]).drop_duplicates(subset=["task", "arm"]))
    sns.set_theme(style="whitegrid", context="talk", font_scale=.84)
    palette = {"control": "#267d95", "glm": "#9a59aa"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), sharex=True, constrained_layout=True)
    task_order = ["promoter", "enhancer", "open_chromatin"]
    for ax, (metric, title) in zip(axes, [("auprc", "Precision–recall AUC"),
                                          ("auroc", "ROC AUC"),
                                          ("f1", "F1, validation-chosen threshold")]):
        sns.barplot(data=unique, x="task", y=metric, hue="arm", order=task_order,
                    hue_order=["control", "glm"], palette=palette, errorbar=None, ax=ax,
                    edgecolor="white", linewidth=1.2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.set_xticks(range(3), ["Promoter", "Enhancer", "Open chromatin"], rotation=12, ha="right")
        ax.grid(axis="y", alpha=.18)
        sns.despine(ax=ax, left=True)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
        if ax is axes[0]:
            handles, _ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    fig.legend(handles, ["Composition only", "+ NT-v2 50M"], loc="upper center",
               bbox_to_anchor=(.5, 1.045), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Agent-selected pipelines · one held-out set per task", fontweight="bold", y=1.14)
    fig.text(.5, -.01, "Three Pi repetitions per arm chose the same pipeline; bars are NOT independent test replicates.",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(ROOT / "comparison_v2.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    validation = json.loads((ROOT / ".cache" / "validation.json").read_text())
    results = [grade(path, validation) for path in sorted((ROOT / "runs").iterdir()) if path.is_dir()]
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    columns = ["run_id", "task", "arm", "trial", "score", "possible", "selected_features", "C", "threshold",
               "duration_seconds", "tool_calls", "tool_errors", "reported_pi_cost_usd", "exit_code", "agent_settled"]
    with (ROOT / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    plot(results)
    for task in ("promoter", "enhancer", "open_chromatin"):
        for arm in ("control", "glm"):
            subset = [r for r in results if r["task"] == task and r["arm"] == arm]
            print(task, arm, [(r["selected_features"], r["score"], r["audit"]["optimal_validation_choice"],
                               r["held_out"]["auprc"] if r["held_out"] else None) for r in subset])


if __name__ == "__main__":
    main()
