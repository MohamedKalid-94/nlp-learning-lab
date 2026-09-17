import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Same embedding model used throughout — consistency matters,
# you can't mix vectors from different models in one DB
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a local ChromaDB client (persists to disk in this example)
client = chromadb.Client()

# Create a collection with explicit cosine distance metric
# (matches what you configured in RAG Day 5)
collection = client.create_collection(
    name="support_docs",
    metadata={"hnsw:space": "cosine"}  # explicitly set distance metric
)

documents = [
    "How to reset your account password step by step",
    "Troubleshooting Bluetooth connection issues",
    "Understanding your monthly billing statement",
    "How to cancel your subscription",
]

# Embed all documents and add them to the collection,
# each needs a unique ID
doc_vectors = embedder.embed_documents(documents)
collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    embeddings=doc_vectors,
    documents=documents,
)

# Query: embed the query, then ask ChromaDB for the top 2 nearest matches
query = "I forgot my login credentials, how do I get back in?"
query_vector = embedder.embed_query(query)

results = collection.query(
    query_embeddings=[query_vector],
    n_results=2,  # top-K
)

print("Top matches:")
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"  distance={distance:.3f} -> {doc}")