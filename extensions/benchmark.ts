import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Only these narrow tools are offered to the agent. No shell, file reader, network,
// arbitrary Python execution, or filesystem-writing tool is registered.
function loadJson(path: string): any {
  const root = process.env.BENCH_ROOT;
  if (!root) throw new Error("BENCH_ROOT is required");
  return JSON.parse(readFileSync(join(root, path), "utf8"));
}

function result(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value) }], details: undefined };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool(defineTool({
    name: "describe_dataset",
    label: "Describe public promoter benchmark",
    description: "Describe the public promoter classification task, sample sizes, and held-out split; no scores.",
    parameters: Type.Object({}),
    async execute() {
      const source = loadJson("data/provenance.json");
      return result({ dataset: source.dataset, source_revision: source.revision,
        task: "Classify 251-base human non-TATA promoter sequences (label 1) versus negatives (label 0)",
        selected_train: 128, selected_test: 64, balanced_classes: true,
        split: "Separate official source train/test splits; deterministic balanced subsets",
        caveat: "Small subset; no chromosome-level or homology-aware split audit" });
    },
  }));

  pi.registerTool(defineTool({
    name: "evaluate_gc_3mer",
    label: "Evaluate composition baseline",
    description: "Return fixed validation and held-out test results for GC content + normalized 3-mer frequencies with logistic regression. Same protocol as the genomic-model feature tool.",
    parameters: Type.Object({}),
    async execute() {
      const evaluation = loadJson(".cache/evaluation.json");
      return result({ protocol: evaluation.protocol, ...evaluation.gc_3mer });
    },
  }));

  if (process.env.BENCH_ARM === "glm") {
    pi.registerTool(defineTool({
      name: "evaluate_nt_embeddings",
      label: "Evaluate Nucleotide Transformer features",
      description: "Return fixed validation and held-out test results using mean-pooled Nucleotide Transformer v2 50M embeddings and the same logistic-regression protocol as the baseline.",
      parameters: Type.Object({}),
      async execute() {
        const evaluation = loadJson(".cache/evaluation.json");
        const model = loadJson("model_provenance.json");
        return result({ protocol: evaluation.protocol, model: model.model,
          model_revision: model.revision, ...evaluation.nt_v2_50m });
      },
    }));
  }
}
