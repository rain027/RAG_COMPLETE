# Multi-Signal RAG for Scientific QA

A retrieval-augmented pipeline for answering questions over long research papers, built and evaluated on [QASPER](https://github.com/allenai/qasper) — a dataset of NLP papers paired with questions that require finding evidence scattered across full-text sections.

Standard RAG setups usually rank chunks by embedding similarity alone. This project explores whether combining several *independent* retrieval signals into a single weighted score gets closer to the evidence a reader would actually reach for.

## Why multi-signal?

Papers aren't flat text — an answer to "what dataset did they use?" is far more likely to sit in the Methodology or Results section than deep in Related Work, and evidence chunks tend to cluster near the beginning or end of a document rather than the middle. The scorer here blends five signals to try to capture that:

| Signal | What it captures | Weight |
|---|---|---|
| Semantic | Cosine similarity between query and chunk embeddings | 0.30 |
| Lexical | BM25 score over raw text | 0.20 |
| Structural | Section-type prior (abstract, results, etc. weighted higher) | 0.25 |
| Position | U-shaped prior favoring chunks near the start/end of a paper | 0.15 |
| Section type | Secondary section-level weighting | 0.10 |

These are combined into a single score per chunk, then used for budget-aware retrieval (chunks are packed into a fixed token budget rather than a fixed top-k).

## What's in here

- `qasper_implementation.py` — dataset loader, chunking/indexing, the multi-signal scorer, and the core retriever + evaluator (Recall@k, Precision@k, MRR, answer coverage)
- `baseline_comparison.py` / `complete_baseline_comparison.py` — compares three retrievers head-to-head: plain semantic-only baseline, semantic + lexical, and the full multi-signal retriever, including a paired t-test for statistical significance
- `download_qasper.py`, `check_qasper_structure.py` — dataset utilities
- `setup_guide.md` — environment setup and usage walkthrough

## Status

This is very much a work in progress, albeit one with a bright future - the multi-signal approach already beats the semantic-only baseline, and the main open question now is finding a more principled (and efficient) way to weight and combine the signals, rather than the current hand-picked values. The current weights on each signal were hand-picked, not learned or tuned systematically — figuring out a more principled (and hopefully more efficient) way to combine or learn these signals is the main thing I'm actively working through right now. Expect the scoring logic, weighting, and possibly the chunking strategy to keep changing as I test against baselines and dig into where retrieval is still missing evidence.

Contributions, ideas, or "have you tried X" suggestions are welcome.
