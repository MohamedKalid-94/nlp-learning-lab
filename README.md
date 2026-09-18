# NLP Learning Lab

A structured, hands-on journey through NLP — from classical text
processing to modern Transformers to a full applied project. Every
topic includes a theory note and a working Python implementation,
run and verified against real output (not just written, but executed).

## Structure

| Phase | Topics | Focus |
|---|---|---|
| [Phase 1: Classical NLP](./Phase-1-Classical-NLP) | 1-10 | Preprocessing, tokenization, stemming, lemmatization, POS tagging, NER, TF-IDF, Bag of Words, text classification |
| [Phase 2: Modern NLP](./Phase-2-Modern-NLP) | 11-21 | Word2Vec, GloVe, FastText, sequence models (RNN/LSTM), language modeling, attention, positional encoding, Transformers, BERT, GPT, evaluation metrics |
| [Phase 3: GenAI](./Phase-3-GenAI) | 22-30 | Embeddings, semantic search, vector databases, chunking, retrieval, reranking, RAG, agentic AI, NLP task overview |
| [Support Ticket Project](./Support-Ticket-Project) | Applied | All 10 classical NLP topics combined into one working, end-to-end system |

Each topic folder contains:
- A note (`.md`) — What it is, why it matters, how it works, a real-world
  example, a Python implementation, interview Q&A, and where it fits in
  the broader GenAI pipeline
- A working implementation (`main.py`) — run and verified, not just
  written

## Capstone Project: Support Ticket Intelligence System

A complete pipeline that takes a raw, messy customer support ticket and
returns a structured summary: category, requested action, and extracted
entities (order numbers, products, dates, amounts).

**Example:**
Input: "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and
it's been 2 weeks. I paid $89.99. Please refund me."

Output:
Category: billing
Action verb: refund
Entities: orders: ['#48213'], money: ['$89.99'],
dates: ['2 weeks'], products: ['the SmartWatch X2']


Built on a real 100-ticket dataset across 10 categories. Uses all 10
classical NLP topics chained into one pipeline: preprocessing →
tokenization → lemmatization → POS tagging (action extraction) → NER
(hybrid spaCy + regex) → TF-IDF vectorization → Naive Bayes
classification.

Honest results: 70% classification accuracy on a held-out test set,
with documented limitations (small dataset means noisy per-category
scores, some category overlap between "billing" and "refund").

See [Support-Ticket-Project/README.md](./Support-Ticket-Project) for
full architecture, setup, and usage.

## Setup

Root-level `requirements.txt` covers dependencies for exploring
Phases 1-3 topic-by-topic. The Support Ticket Project has its own
self-contained `requirements.txt` — see its README for setup specific
to running the capstone.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Why this repo exists

Built as a structured learning path while transitioning into AI/ML —
covering the full evolution of NLP from word counting (TF-IDF) to
transformer attention (BERT/GPT) to applied retrieval-augmented
systems (RAG), with every concept backed by working code, not just notes.

git push

Want the Phase-2-Modern-NLP/README.md mini-version next, or the LICENSE file?
