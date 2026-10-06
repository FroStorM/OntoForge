# OntoForge

> OntoForge: Continual Ontology and Knowledge Graph Construction via Multi-Agent Consensus and Self-Refining Procedures

## Contents

- `OntoForge.tex`: LaTeX manuscript source
- `OntoForge.bbl`: Compiled bibliography used by the manuscript
- `references.bib`: Bibliography source
- `figure1-v2.png`: Framework overview figure
- `figure2.png`: Sequential evaluation figure
- `OntoForge_CIKM2026_CR.pdf`: Final compiled PDF
- `data/`: Lightweight sample data and the exact reported-run sample manifest
- `experiments/`: Reproduction scripts, protocols, and final experiment data

The complete Text2KGBench dataset is not redistributed here. See
`data/README.md` and `experiments/docs/experiment_protocol.md` for the
official dataset setup.

## Build

Compile with a current LaTeX distribution and the ACM `acmart` class:

```bash
pdflatex OntoForge.tex
bibtex OntoForge
pdflatex OntoForge.tex
pdflatex OntoForge.tex
```

The source expects `figure1-v2.png` and `figure2.png` in the same directory.

## Experiments

The final reported result files are under `experiments/results/`. The four
main backbone runs and the completed Mimo ablation are documented in
`experiments/docs/results_index.md`.
