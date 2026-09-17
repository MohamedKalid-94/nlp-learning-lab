from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# Setup: chunk + embed + store (done once, ahead of time)
chunks = [
    "Week 1: Introduction to Python basics and data types.",
    "Week 2: Object-oriented programming, classes and inheritance.",
    "Week 3: Web frameworks, Flask basics and REST APIs.",
]
chunk_vectors = embedder.embed_documents(chunks)
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=chunk_vectors,
    documents=chunks,
)

def retrieve(query: str, top_k: int = 2) -> list[str]:
    """
    The one function that ties embeddings + vector DB + semantic search
    together: takes a raw query, returns the top-K most relevant chunks.
    This is the exact interface your RAG project's retriever.py exposes.
    """
    query_vector = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results["documents"][0]

# Test it
query = "How do I build a REST API?"
relevant_chunks = retrieve(query)

print(f"Query: {query}\n")
print("Retrieved chunks:")
for chunk in relevant_chunks:
    print(f"  - {chunk}")