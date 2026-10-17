import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { promisify } from "node:util";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const execFileAsync = promisify(execFile);
const tasks = ["promoter", "enhancer", "open_chromatin"];
const cs = [0.01, 0.1, 1.0];
const thresholds = [0.35, 0.5, 0.65];

function output(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value) }], details: undefined };
}

export default function (pi: ExtensionAPI) {
  const root = process.env.BENCH_ROOT;
  const task = process.env.BENCH_TASK;
  const arm = process.env.BENCH_ARM;
  if (!root || !task || !tasks.includes(task) || (arm !== "control" && arm !== "glm")) {
    throw new Error("BENCH_ROOT, BENCH_TASK and BENCH_ARM must be set by the runner");
  }
  const available = arm === "glm" ? ["gc_3mer", "gc_5mer", "nt_v2_50m"] : ["gc_3mer", "gc_5mer"];
  const allowedFeatures = Type.Union(available.map((name) => Type.Literal(name)) as [any, ...any[]]);
  const options = Type.Object({
    features: allowedFeatures,
    C: Type.Union([Type.Literal(0.01), Type.Literal(0.1), Type.Literal(1)]),
    threshold: Type.Union([Type.Literal(0.35), Type.Literal(0.5), Type.Literal(0.65)]),
  });
  const validation = JSON.parse(readFileSync(join(root, ".cache/validation.json"), "utf8"));
  const provenance = JSON.parse(readFileSync(join(root, "data/provenance.json"), "utf8"));
  let submitted = false;

  pi.registerTool(defineTool({
    name: "describe_task", label: "Describe sequence task",
    description: "Get the assigned public DNA classification task, permitted features, validation protocol, and sample sizes.",
    parameters: Type.Object({}),
    async execute() {
      const meta = provenance.datasets[task];
      return output({ task, source: meta.repository, revision: meta.revision,
        train_rows: meta.subsets.train.rows, held_out_rows: meta.subsets.test.rows,
        sequence_lengths: [meta.subsets.train.length_min, meta.subsets.train.length_max],
        allowed_features: available, validation_protocol: validation.protocol,
        caution: "The source provides train/test splits, but genomic coordinates and homology-aware split verification are unavailable." });
    },
  }));

  pi.registerTool(defineTool({
    name: "validate_feature", label: "Run training-only validation",
    description: "Get all training-validation scores (C x threshold grid) for one allowed feature family; NO test metrics.",
    parameters: Type.Object({ features: allowedFeatures }),
    async execute(_id, args) {
      if (!available.includes(args.features)) throw new Error("Feature not available in this arm");
      return output({ task, features: args.features, configurations: validation.tasks[task][args.features] });
    },
  }));

  pi.registerTool(defineTool({
    name: "submit_pipeline", label: "One blind held-out test submission",
    description: "FINALIZE one feature family, regularization C and threshold. Refit on all training data and return held-out metrics. One use per run; choose using validation only.",
    parameters: options,
    async execute(_id, args) {
      if (submitted) throw new Error("Held-out test has already been submitted in this run");
      if (!available.includes(args.features) || !cs.includes(args.C) || !thresholds.includes(args.threshold)) {
        throw new Error("Invalid or unavailable configuration");
      }
      submitted = true;
      const python = join(root, "..", ".venv", "bin", "python");
      const script = join(root, "scripts", "submit.py");
      const { stdout } = await execFileAsync(python,
        [script, task, args.features, String(args.C), String(args.threshold)],
        { cwd: root, timeout: 60_000, maxBuffer: 1024 * 1024 });
      return output(JSON.parse(stdout));
    },
  }));
}
