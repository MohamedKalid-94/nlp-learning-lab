from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

# Same LLM setup pattern as your RAG project's generator.py
llm = ChatGroq(model="openai/gpt-oss-20b")

# 1. Sentiment Analysis
sentiment_prompt = "Classify the sentiment as positive, negative, or neutral: 'This product completely changed how I work, love it!'"
print("Sentiment:", llm.invoke(sentiment_prompt).content)

# 2. Summarization
long_text = """Machine learning is a subset of artificial intelligence that enables
systems to learn from data rather than being explicitly programmed. It has
applications across industries including healthcare, finance, and transportation."""
summary_prompt = f"Summarize this in one sentence: {long_text}"
print("\nSummary:", llm.invoke(summary_prompt).content)

# 3. Translation
translate_prompt = "Translate to French: 'How do I reset my password?'"
print("\nTranslation:", llm.invoke(translate_prompt).content)

# 4. Question Answering (using context, exactly like your RAG pipeline)
qa_prompt = """Context: The store is open Monday to Friday, 9am to 6pm.
Question: What time does the store open on Saturdays?
Answer using only the context above."""
print("\nQA:", llm.invoke(qa_prompt).content)