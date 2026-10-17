# Experimental design, grading, and limitations

## Question and estimand

Does giving a Pi agent access to **Nucleotide Transformer v2 50M sequence embeddings** improve its answer to a small promoter-classification research task relative to access to ordinary composition features alone? This first iteration measures the effect of **feature-tool availability in a constrained workflow**. It does **not** measure a general-purpose coding agent with unrestricted shell access, end-to-end discovery, statistical significance of a genomic model improvement, or performance across biology research broadly.

## Public data and deterministic subset

Dataset: [Genomic Benchmarks human non-TATA promoters](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_nontata_promoters), upstream Git revision `e0003669df0a180c8570ebe091ed91f20fa080ec`. The original official train and test Parquet files are downloaded and SHA-256 checked against `data/provenance.json`. `prepare_data.py` samples 64 label-0 and 64 label-1 training sequences, plus 32 of each from the official test split, with pinned seed `20261017`. All are 251 nucleotides. The dataset card lacks a stated redistribution license, so downloaded and sampled sequences are ignored by Git and must be fetched locally.

We check exact sequence duplicates across the selected train/test sets, but **do not** audit reverse complements, nearby genomic loci, sequence homology, chromosome overlap, or species/gene-family leakage. These could inflate evaluation metrics. This is a demonstration, not a publishable estimate of generalization.

## Features and fixed classifier protocol

- **Control:** GC fraction + normalized 3-mer frequencies (65 dimensions).
- **Treatment:** mean pooling the last hidden state of [Nucleotide Transformer v2 50M multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) with attention masking (512 dimensions). Model revision and downloaded code/weight checksums are in `model_provenance.json`; embedding device/runtime details are in `embedding_provenance.json`. The model is used frozen, without fine-tuning.
- Both representations use `StandardScaler` followed by `LogisticRegression`. The same stratified 25% slice of the training subset selects `C` from `[0.01, 0.1, 1, 10]` by validation AUROC, then validation accuracy, then smallest `C`. Refit on the complete training subset; evaluate the held-out test subset once. This is implemented in `scripts/evaluate_features.py` and cached in `.cache/evaluation.json` before any agent run.
- **Agent choice:** each agent sees only tools in its arm. It selects the highest *training-validation* AUROC among available representations, then reports *test* AUROC and threshold-0.5 accuracy. These are intentionally distinct metrics; better ranking can coexist with worse classification at a fixed threshold.

The cached evaluation results are fixed for all five trials per arm. Repeated Pi calls measure answer/tool-use consistency, **not five independent biological experiments or five independent test sets**. Never treat ten agent responses as ten independent estimates of classifier performance.

## Tool-access control

`scripts/run_trials.py` invokes Pi with `--no-extensions --extension extensions/benchmark.ts --no-mcp --no-skills --no-prompt-templates --no-context-files --no-themes --no-approve` and an exact `--tools` allowlist. Neither arm gets `bash`, `read`, `edit`, `write`, browsing, arbitrary network access, or an agent-controlled code runner. The control registers only `describe_dataset` and `evaluate_gc_3mer`; the treatment additionally registers `evaluate_nt_embeddings`. The extension reads private cached results from the host, not from agent-visible tools. Unlike a prompt-only prohibition, the control has no registered GLM operation. This is **tool-level isolation, not an OS security boundary against a malicious extension or Pi process**. Use a container/VM and network rules before allowing arbitrary agent code in a stricter follow-up.

Both arms use `openai/gpt-6-sol`, medium thinking, the same single prompt, and separate persistent Pi sessions. The one extra tool's declaration and output also add prompt tokens; tool availability is the treatment as a whole, not an isolated causal effect of the embedding weights. No labels or held-out predictions are passed directly in the agent prompt.

## Scoring and traces

`scripts/score_and_plot.py` parses the **last finalized assistant message** from each Pi JSON event stream as a JSON object. Seven exact/numeric fields are graded: selected representation and six baseline/selected/delta AUROC/accuracy values. Numeric tolerance: `rel_tol=1e-4`, `abs_tol=1e-8`. Interpretations and caveats are retained for human review, but **not semantically graded**. Usage-based Pi model cost, wall time, tool calls, retries, and the full event/session traces are also saved. The reference results are computed before Pi runs and cached privately; Pi agents cannot access the evaluator or its data through registered tools.

Each `runs/<arm>-trial-NN/` directory contains `events.jsonl`, `stderr.log`, `manifest.json`, and a `sessions/` subtree with Pi's JSONL session. Session entries have IDs and parent IDs; the manifest pairs control and treatment run IDs. Event records include observable tool decisions and responses. Provider-private reasoning is not recoverable from traces.

## Pilot observation

| Available features | Validation AUROC | Test AUROC | Test accuracy |
|---|---:|---:|---:|
| GC + 3-mers | 0.917969 | 0.886719 | 0.859375 |
| NT-v2 50M | 0.968750 | 0.916016 | 0.812500 |

Both arms returned 7/7 numeric/selection fields in all five agent repetitions. Tool availability led the treatment agents to select embeddings based on validation AUROC. On the **same fixed 64-sequence test subset**, this representation gained `+0.029297` AUROC but lost `-0.046875` accuracy (three fewer correct classifications at the 0.5 cutoff). No significance test or uncertainty interval has been calculated for these differences. The latter may reflect calibration or threshold choice; we did not investigate that here.

Next steps for a stronger uplift claim: multiple datasets and task types; homology/chromosome-aware splits; bootstrap confidence intervals by test sequence (with dependence caveats); independent sample-level tasks; equal analysis/compute budgets; and a code-execution sandbox that still prevents model access in the control.
