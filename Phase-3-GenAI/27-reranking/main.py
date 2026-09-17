from sentence_transformers import CrossEncoder

# Cross-encoder model — same one used in your RAG Day 12 implementation
# Unlike the bi-encoder (topic 22), this model looks at query+document
# TOGETHER, not as separate precomputed vectors
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How do I build a REST API?"

# Pretend these are the candidates retrieval already narrowed down to
# (in a real pipeline, this list would come from topic 26's retrieve() function)
candidates = [
    "Week 3: Web frameworks, Flask basics and REST APIs.",
    "Week 2: Object-oriented programming, classes and inheritance.",
    "Week 1: Introduction to Python basics and data types.",
]

# Cross-encoder needs [query, candidate] PAIRS as input —
# it scores each pair jointly, not independently like embeddings do
pairs = [[query, candidate] for candidate in candidates]
scores = reranker.predict(pairs)

# Sort candidates by their new cross-encoder score, highest first
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

print(f"Query: {query}\n")
print("Reranked results:")
for candidate, score in ranked:
    print(f"  {score:.3f} -> {candidate}")