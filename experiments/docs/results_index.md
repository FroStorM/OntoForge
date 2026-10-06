# Results Index

This file tracks which result directories should be included in the final anonymous supplementary package.

## Included in Main Paper

| Backbone | Setting | Status | Source |
| --- | --- | --- | --- |
| GPT-5.5 | Wikidata-TEKGEN, four domains, limit 50/domain | completed | `results/text2kg-GPT-5.5-wikidata4-limit50/metrics.csv` |
| DeepSeek-v4-pro | Wikidata-TEKGEN, four domains, limit 50/domain | completed | `results/text2kg-deepseek-wikidata4-limit50/metrics.csv` |
| Mimo-v2.5-pro | Wikidata-TEKGEN, four domains, limit 50/domain | completed | `results/text2kg-mimo-wikidata4-limit50/metrics.csv` |
| Qwen-3.6-plus | Wikidata-TEKGEN, four domains, limit 50/domain | completed | `results/text2kg-qwen-wikidata4-limit50/metrics.csv` |

## Files To Include Per Completed Run

For each reported backbone, include:

- `metrics.csv`
- `sample_manifest.json`
- `predictions_llm_only.jsonl`
- `predictions_llm_sop.jsonl`
- `predictions_ontoforge.jsonl`

Before packaging, inspect files for API keys, local account names, private URLs, and non-anonymous metadata.

The canonical sample manifest for the main results is also provided as `manifests/sample_manifest_wikidata4_limit50_seed13.json`.
