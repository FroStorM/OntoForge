# Experiments

This directory contains the scripts, protocol documents, sampling manifest,
and final result files used for the paper experiments.

## Final Results

The `results/` directory contains:

- Four main runs on Wikidata-TEKGEN with four domains and 50 instances per
  domain:
  - GPT-5.5
  - DeepSeek-v4-pro
  - Mimo-v2.5-pro
  - Qwen-3.6-plus
- The completed Mimo-v2.5-pro component ablation run.

Each main run includes metrics, the sample manifest, predictions for the
LLM-only, static-SOP, and OntoForge settings, and the generated result table.
The ablation run includes the corresponding ablation predictions and metrics.

See `docs/results_index.md` for the canonical mapping between paper results
and result files. API credentials, local model configuration, caches, and
machine-specific logs are intentionally excluded.
