from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Same embedding model as topic 22 — must use the SAME model for
# both documents and queries, otherwise vectors aren't comparable
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pretend this is a small knowledge base of support articles
documents = [
    "How to reset your account password step by step",
    "Troubleshooting Bluetooth connection issues",
    "Understanding your monthly billing statement",
    "How to cancel your subscription",
]

# Embed all documents ONCE — in a real system you'd store these
# in a vector database (topic 24) instead of recomputing every search
doc_vectors = embedder.embed_documents(documents)

# A user query phrased completely differently from the matching document,
# on purpose — this is the real test of semantic vs keyword search
query = "I forgot my login credentials, how do I get back in?"
query_vector = embedder.embed_query(query)  # embed_query, not embed_documents, for a single text

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Score the query against every document, then rank by similarity
scores = [cosine_sim(query_vector, doc_vec) for doc_vec in doc_vectors]
ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

print("Query:", query)
print("\nRanked results (most relevant first):")
for doc, score in ranked:
    print(f"  {score:.3f} -> {doc}")