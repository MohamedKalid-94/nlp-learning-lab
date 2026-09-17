from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# Setup: chunk (simplified, already pre-chunked here) + embed + store
chunks = [
    "Week 1: Introduction to Python basics and data types.",
    "Week 2: Object-oriented programming, classes and inheritance.",
    "Week 3: Web frameworks, Flask basics and REST APIs.",
]
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embedder.embed_documents(chunks),
    documents=chunks,
)

def retrieve(query: str, top_k: int = 2) -> list[str]:
    query_vector = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results["documents"][0]

def build_prompt(query: str, context_chunks: list[str]) -> str:
    """
    This is the 'Augmented' step: inserting retrieved context directly
    into the prompt, with an explicit instruction to stay grounded in it
    (the same grounding principle your generator.py Day 9 used)
    """
    context = "\n".join(context_chunks)
    return f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

# Full RAG flow: retrieve -> augment -> (generate, shown as a print here
# instead of a real LLM call, since that needs an API key)
query = "What will I learn about building APIs?"
relevant_chunks = retrieve(query)
final_prompt = build_prompt(query, relevant_chunks)

print("=== Retrieved Context ===")
for chunk in relevant_chunks:
    print(" -", chunk)

print("\n=== Final Prompt sent to LLM ===")
print(final_prompt)