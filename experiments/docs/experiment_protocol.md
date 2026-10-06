# Experiment Protocol

## Dataset

We evaluate on Text2KGBench using the Wikidata-TEKGEN split. The main experiments use four domains:

- `1_movie`
- `2_music`
- `6_computer`
- `8_politics`

For each domain, we sample 50 test instances with seed `13`, resulting in 200 instances per LLM backbone. During debugging or model screening, smaller limits such as 5 or 20 instances per domain may be used, but the main paper reports the 50-instance-per-domain setting unless otherwise stated.

The exact sampled Text2KGBench instances used in the main experiments are provided in `manifests/sample_manifest_wikidata4_limit50_seed13.json`. The reported model backbones share this same sample manifest.

## Methods

We compare three ontology-aware construction variants under the same LLM backbone:

- **LLM-only**: direct ontology-guided triple extraction with a generic schema prompt.
- **LLM+SOP**: single-pass extraction augmented with static domain-specific SOP instructions.
- **OntoForge**: multi-agent candidate generation with consensus verification and adaptive SOP memory.

## Leakage Control

Text2KGBench provides official prompt examples. To avoid leakage, any official demonstration whose example sentence exactly matches the test sentence is disabled. Safe demonstrations may still be used for output-format and normalization guidance.

## LLM Calls

The runner uses an OpenAI-compatible chat-completion API. API credentials are kept outside the supplementary package. Calls use JSON-only responses, deterministic temperature, local caching, and retry handling.

Default retry policy:

- maximum retries: 3;
- request sleep: 3 seconds for main runs;
- retry sleep: 10 seconds with exponential backoff.

## Metrics

We report triple-level Precision, Recall, and F1 after normalizing subject strings, relation labels, and object strings. A predicted triple is counted as correct only when the normalized subject, relation, and object all match a gold triple.

Ontology Conformance is computed as:

```text
number of predicted triples whose relation is defined in the ontology
--------------------------------------------------------------------
number of predicted triples
```

Grounding is computed as the fraction of predicted triples whose subject or object surface strings are grounded in the input sentence after normalization. These diagnostic metrics are used to interpret errors but are not the main ranking criterion.

## Main Run Command

```bash
python scripts/text2kg_experiment.py \
  --model-profile MODEL_PROFILE \
  --dataset wikidata_tekgen \
  --domains 1_movie 2_music 6_computer 8_politics \
  --limit-per-domain 50 \
  --seed 13 \
  --request-sleep 3 \
  --retry-sleep 10 \
  --out-dir results/text2kg-MODEL_PROFILE-wikidata4-limit50
```
