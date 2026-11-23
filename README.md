# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py            # Run TF-IDF on FIQA
python main.py top_k=20   # Override a config value
```

This prints recall@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```
