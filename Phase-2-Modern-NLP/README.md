# Phase 2: Modern NLP

Word-level and architecture-level foundations behind every modern LLM —
from early embedding methods through the Transformer architecture that
powers BERT, GPT, and beyond.

| # | Topic | What it covers |
|---|---|---|
| [11](./11-word2vec) | Word2Vec | Word embeddings via context prediction (CBOW/Skip-gram) |
| [12](./12-glove) | GloVe | Word embeddings via global co-occurrence statistics |
| [13](./13-fasttext) | FastText | Word embeddings robust to unseen words via character n-grams |
| [14](./14-sequence-models) | Sequence Models (RNN/LSTM) | Order-aware processing; the pre-Transformer approach |
| [15](./15-language-modeling) | Language Modeling & Perplexity | The next-word-prediction objective behind GPT, and how to measure it |
| [16](./16-attention) | Attention | The mechanism letting every word "see" every other word directly |
| [17](./17-positional-encoding) | Positional Encoding | Injecting word order back into attention-based models |
| [18](./18-transformers) | Transformers | The full architecture: attention + positional encoding + feed-forward, stacked |
| [19](./19-bert) | BERT | Encoder-only, bidirectional — built for understanding tasks |
| [20](./20-gpt) | GPT | Decoder-only, causal — built for text generation |
| [21](./21-evaluation-metrics) | Evaluation Metrics | F1, BLEU, ROUGE, Perplexity — how to measure model quality |

**Conceptual arc**: 
Word2Vec/GloVe/FastText (topics 11-12, 21) solve
"turn a word into a vector." 
Sequence Models (13) add order-awareness.
Attention (15) removes the need to process words sequentially.
Positional Encoding (16) reintroduces order without sequential
processing. Transformers (17) combine all of this into one
architecture. BERT (18) and GPT (19) are the two dominant named models built on that architecture — one for understanding, one for generating. 
Evaluation Metrics (20) covers how any of this gets measured objectively.