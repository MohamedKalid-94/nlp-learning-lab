from langchain_text_splitters import RecursiveCharacterTextSplitter

# A fake multi-paragraph document to chunk
document = """
Week 1: Introduction to Python basics, variables, and data types.
This week covers fundamental syntax and simple exercises.

Week 2: Object-oriented programming concepts including classes,
inheritance, and polymorphism. Practice building small class hierarchies.

Week 3: Introduction to web frameworks, focusing on Flask basics
and building simple REST APIs for beginners.
"""

# chunk_size = max characters per chunk
# chunk_overlap = characters shared between consecutive chunks,
# so context isn't lost right at a chunk boundary
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_text(document)

print(f"Number of chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.strip())
    print()