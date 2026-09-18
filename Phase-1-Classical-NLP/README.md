# Phase 1: Classical NLP

The foundational text-processing pipeline used before deep learning
took over NLP — still relevant for interviews, lightweight systems,
and understanding *why* modern methods exist.

| # | Topic | What it covers |
|---|---|---|
| [01](./01-text-processing) | Text Preprocessing | Cleaning raw text: lowercasing, punctuation/URL/HTML removal, whitespace normalization |
| [02](./02-tokenization) | Tokenization | Splitting text into words — regex, NLTK, spaCy, and Hugging Face compared |
| [03](./03-subword-tokenization) | Subword Tokenization | BPE/WordPiece — how modern LLM tokenizers actually work |
| [04](./04-stemming) | Stemming | Blind rule-based word-ending removal (Porter Stemmer) |
| [05](./05-lemmatization) | Lemmatization | Dictionary + grammar-aware normalization to real base words |
| [06](./06-pos-tagging) | POS Tagging | Labeling each word's grammatical role (noun, verb, etc.) |
| [07](./07-ner) | NER | Extracting real-world entities: people, orgs, dates, money |
| [08](./08-tfidf) | TF-IDF | Scoring word importance: frequent here, rare elsewhere |
| [09](./09-bag-of-words) | Bag of Words | Raw word-count vectors, the representation TF-IDF builds on |
| [10](./10-text-classification) | Text Classification | Using TF-IDF + Naive Bayes to actually decide a category |

**Conceptual arc**: 
Preprocessing and Tokenization (01-03) turn raw text into clean, split units. 
Stemming and Lemmatization (04-05) normalize word forms.
POS Tagging and NER (06-07) extract grammatical and real-world structure.
TF-IDF and Bag of Words (08-09) turn text into numbers.
Text Classification (10) uses those numbers to make an actual decision — tying the whole phase together.

## Applied Project

All 10 topics here are combined into one real, end-to-end system in
[Support-Ticket-Project](../Support-Ticket-Project) — a support ticket
classifier that cleans, tags, extracts entities from, and categorizes
real customer tickets.