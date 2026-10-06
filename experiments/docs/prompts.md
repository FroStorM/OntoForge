# Prompt Templates

This file summarizes the prompt templates used by the experiment runner. Full executable templates are implemented in `scripts/text2kg_experiment.py`.

## LLM-only

The LLM-only variant receives the ontology concepts, ontology relations, one safe example when available, and the test sentence. It is instructed to extract triples according to the ontology and return strict JSON.

Core instruction:

```text
Extract triples from the test sentence according to the relations in the ontology.
Only include triples in the required JSON output format.
```

## LLM+SOP

The LLM+SOP variant receives the same ontology and sentence, but additionally receives a static domain SOP.

Static SOP:

```text
1. First identify subject and object mentions in the sentence.
2. Match each evidence phrase to one ontology relation.
3. Reject triples whose relation is outside the ontology.
4. Avoid adding background knowledge not stated in the sentence.
5. Normalize relation names to the exact ontology labels.
```

## OntoForge

OntoForge uses four candidate-generation agents followed by a consensus verifier.

Candidate agents:

- **Recall-oriented relation agent**: proposes all plausible ontology triples directly supported by the sentence.
- **Precision-oriented schema agent**: proposes only high-confidence schema-compatible triples.
- **Canonicalization and alias-repair agent**: repairs subject/object surface forms and aliases.
- **SOP-guided domain agent**: applies static SOPs and adaptive SOP memory to recover missed triples.

Consensus verifier:

```text
1. Select or minimally repair candidate triples; do not invent new triples outside the candidate list.
2. Keep triples whose relation is in the ontology and whose evidence phrase directly supports the relation.
3. Include all distinct supported relations and coordinated objects.
4. Prefer precision or canonicalization candidates, while allowing recall-only candidates with explicit evidence.
5. Normalize subject/object strings using safe examples when available.
6. Remove duplicates, generic placeholders, and unsupported hallucinations.
7. Add SOP updates for recurring mistakes, missing aliases, or relation confusions.
```

The adaptive SOP memory stores reusable correction rules from previous examples and is passed to later OntoForge agents.
