# Phase 3: GenAI

How modern LLM applications actually get built — the retrieval and
agentic-reasoning layer that sits on top of the Transformer models
from Phase 2. This phase closes the loop between theory and the
applied [RAG project](../../) built alongside this repo.

| # | Topic | What it covers |
|---|---|---|
| [22](./22-embeddings) | Embeddings | Turning text (sentences/chunks) into meaning-capturing vectors |
| [23](./23-semantic-search) | Semantic Search | Ranking documents by meaning-similarity, not keyword overlap |
| [24](./24-vector-database) | Vector Database | Storing and searching embeddings at scale (ChromaDB) |
| [25](./25-chunking) | Chunking | Splitting documents into focused pieces before embedding |
| [26](./26-retrieval) | Retrieval | Combining chunking + embeddings + vector DB into one `retrieve()` function |
| [27](./27-reranking) | Reranking | Sharpening retrieval's top results with a slower, more accurate cross-encoder |
| [28](./28-rag) | RAG | The full retrieve → augment → generate pipeline |
| [29](./29-agentic-ai) | Agentic AI | Adding retry loops and self-correction on top of RAG |
| [30](./30-nlp-tasks-overview) | NLP Tasks Overview | Sentiment, summarization, translation, QA — the real jobs this all supports |

**Conceptual arc**:
Chunking (25) and Embeddings (22) prepare and represent text.
Vector Database (24) and Semantic Search (23) store and find it.
Retrieval (26) combines those into one callable function.
Reranking (27) sharpens the results. RAG (28) wraps retrieval + generation into a grounded answer. Agentic AI (29) adds decision-making and retries on top.
NLP Tasks Overview (30) names the real-world jobs — sentiment, summarization, translation, QA — that all
of this ultimately serves.

**Note**: every topic here is a formalized, smaller-scale version of
what's already built and running in the accompanying 14-day RAG
project (multi-PDF research assistant with hybrid search, reranking,
and a 5-node agentic graph) — this phase documents the theory against
real, working code you already built.