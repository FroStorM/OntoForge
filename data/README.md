# Data

This directory contains lightweight public data files for inspection and
reproduction:

- `text2kgbench_sample.json`: small Text2KGBench-format example
- `text2kg_ground_truth.json`: local ground-truth reference used during
  development
- `sample_manifest_wikidata4_limit50_seed13.json`: the exact sample manifest
  used for the reported four-domain experiments

The complete Text2KGBench dataset is not copied into this repository. Download
it from the official source and set `TEXT2KG_ROOT` as described in
`../experiments/docs/experiment_protocol.md`.
