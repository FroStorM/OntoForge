# OntoForge: Continual Ontology and Knowledge Graph Construction via Multi-Agent Consensus and Self-Refining Procedures

## Abstract

Ontology-aware knowledge graph (KG) construction must handle ambiguous evidence, heterogeneous schemas, and recurring extraction errors. A fixed prompt may miss valid triples, apply inconsistent entity canonicalization, or fail to improve as construction proceeds.

We present **OntoForge**, a multi-agent framework that jointly refines KG construction and the procedure used to perform it. Specialized agents generate complementary candidates, an evidence- and schema-aware verifier resolves duplicates and conflicts, and a self-refining Standard Operating Procedure (SOP) memory turns recurring issues into reusable rules. The resulting workflow improves triple-level F1 over direct prompting and static-SOP baselines across several LLM backbones, primarily through higher recall. Ablations and sequential diagnostics indicate that adaptive SOP memory contributes beyond additional model calls and can benefit later construction windows after an initial cold-start period.

## Method

For each document, OntoForge uses four task-specific candidate-generation roles:

- **Recall-oriented agent** broadens candidate coverage.
- **Schema-oriented agent** prioritizes ontology-compatible claims.
- **Canonicalization agent** resolves aliases and generic entity mentions.
- **SOP-guided agent** applies procedural rules learned from earlier construction feedback.

A consensus verifier normalizes candidates, groups competing claims, and accepts a triple only when its relation is defined by the ontology, its evidence supports the complete subject-relation-object claim, and it is neither a duplicate nor a generic placeholder. Resolved conflicts and recurring errors can update SOP memory with concise, non-duplicate rules. These procedural updates use internal feedback rather than test gold labels; direct evidence and ontology constraints take precedence over stored rules.

OntoForge maintains two forms of state: accepted triples accumulate in an ontology-specific KG, while SOP memory carries procedural adaptations across construction steps. In the experiments reported here, construction begins with an empty KG, and the ontology schema is fixed within each domain.

## Experimental Evaluation

We evaluate on the Wikidata-TEKGEN split of Text2KGBench, using four domains: movie, music, computer, and politics. Each domain contributes 50 test instances selected with a fixed seed, for 200 instances per backbone. All methods use the same sample manifest, schemas, and evaluation pipeline.

We compare OntoForge with direct ontology-guided extraction (**LLM-only**) and extraction using static SOP instructions (**LLM+SOP**) on GPT-5.5, DeepSeek-v4-pro, Mimo-v2.5-pro, and Qwen-3.6-plus. Performance is measured by triple-level precision, recall, and F1; a triple is correct only when its normalized subject, relation, and object match the gold triple.

| Backbone | LLM-only P/R/F1 | LLM+SOP P/R/F1 | OntoForge P/R/F1 |
| --- | ---: | ---: | ---: |
| GPT-5.5 | 0.350 / 0.373 / 0.361 | 0.315 / 0.339 / 0.326 | **0.375 / 0.443 / 0.406** |
| DeepSeek-v4-pro | 0.356 / 0.387 / 0.371 | 0.353 / 0.356 / 0.354 | 0.327 / **0.433** / **0.373** |
| Mimo-v2.5-pro | 0.341 / 0.242 / 0.283 | **0.360** / 0.269 / 0.308 | 0.342 / **0.366** / **0.353** |
| Qwen-3.6-plus | **0.362** / 0.397 / 0.379 | 0.354 / 0.380 / 0.366 | 0.355 / **0.467** / **0.403** |

OntoForge achieves the highest F1 for all four evaluated backbones, with its clearest gains driven by recall. Precision remains an important trade-off: for DeepSeek-v4-pro, OntoForge raises recall while slightly lowering precision relative to the direct baseline.

On Mimo-v2.5-pro, the component ablation reports F1 of **0.353** for full OntoForge, **0.333** without adaptive SOP, and **0.283** for LLM-only. A five-call self-consistency baseline does not match the full system, suggesting that the gains are not explained by the larger call budget alone. In a sequential diagnostic, OntoForge trails LLM-only in the first cold-start window, then outperforms both LLM-only and static SOP in each subsequent window.

## Scope and Limitations

Text2KGBench does not provide explicit temporal-drift annotations. The sequential setup is therefore a controlled approximation to continual construction, not a definitive lifelong-learning evaluation. This study does not evaluate schema evolution, fact replacement, catastrophic forgetting, or global graph consistency after many updates. The comparisons focus on controlled prompting baselines rather than an exhaustive set of KG, retrieval, entity-linking, or ontology-learning systems. Reported results should be interpreted in light of the sampled evaluation and the absence of statistical significance testing.

## Citation

```bibtex
@inproceedings{guo2026ontoforge,
  author = {Fang Guo and Li Zhu and Mengying Wu},
  title = {OntoForge: Continual Ontology and Knowledge Graph Construction via Multi-Agent Consensus and Self-Refining Procedures},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year = {2026},
  doi = {10.1145/3799682.3839852}
}
```
