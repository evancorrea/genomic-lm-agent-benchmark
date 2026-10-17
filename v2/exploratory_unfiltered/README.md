# Archived exploratory run — NOT the final v2 benchmark

This directory preserves the first 18 v2 Pi runs, their per-run sessions and JSON event traces, results, provenance, plot, and similarity audit. A post-run 7-mer Jaccard screen found **two promoter test sequences with nearest sampled-train similarity ≥ 0.8**, including one ≥ 0.9, despite zero exact/reverse-complement duplicates. No other task had a sampled train/test pair above 0.8.

We did **not** discard these records. Instead, we changed `v2/scripts/prepare_data.py` to select held-out sequences from the same pinned source revision only when their nearest sampled-training 7-mer Jaccard similarity (considering reverse complements) is below 0.8, rebuilt embeddings, and ran a fresh set of 18 Pi sessions in `v2/runs/`. This choice was made **after seeing initial outcomes**, so even the revised subset should be interpreted as a development/pilot result, not a confirmatory test. A new untouched chromosome-aware external dataset is needed for confirmation.

The corresponding private sampled CSVs and cached embeddings for this archive remain locally in `v2/.cache/exploratory_unfiltered/` and are Git-ignored because the source dataset card does not provide redistribution terms. The archived metadata and traces remain available to inspect decisions and lineage.
