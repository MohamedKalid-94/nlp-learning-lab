from langchain_huggingface import HuggingFaceEmbeddings

# Load a lightweight sentence-transformer model (BERT-family descendant)
# 384-dimensional output, runs locally, no API key needed
# This is the exact model you used in RAG Day 4
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Three test sentences:
# - texts[0] and texts[1] are semantically related (both about login/password issues)
#   but share almost NO exact words in common
# - texts[2] is completely unrelated (different topic entirely)
texts = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What's the weather today?",
]

# Convert all 3 sentences into embedding vectors in one call
# Returns a list of vectors, one per input text
vectors = embedder.embed_documents(texts)

# Sanity check: how many vectors did we get, and how many dimensions each?
print("Number of vectors:", len(vectors))      # should be 3 (one per sentence)
print("Dimension per vector:", len(vectors[0]))  # should be 384 (model's output size)

import numpy as np

def cosine_sim(a, b):
    # Cosine similarity = how aligned two vectors are in direction,
    # ignoring their magnitude. Range: -1 (opposite) to 1 (identical direction)
    # This is the standard way to compare embedding vectors for similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare sentence 1 vs sentence 2 (semantically related, different words)
# Expect a HIGH similarity score here, proving embeddings capture meaning,
# not just word overlap
print("Sim(1,2) [related]:", cosine_sim(vectors[0], vectors[1]))

# Compare sentence 1 vs sentence 3 (completely unrelated topics)
# Expect a LOW similarity score here, as a contrast/control check
print("Sim(1,3) [unrelated]:", cosine_sim(vectors[0], vectors[2]))