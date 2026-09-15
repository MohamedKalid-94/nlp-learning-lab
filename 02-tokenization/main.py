text = "I don't like it."

# 1. Regex (manual, for understanding mechanics)
import re
print("Regex :", re.findall(r"\w+|[^\w\s]", text))

# 2. NLTK (classical NLP standard)
from nltk.tokenize import word_tokenize
print("NLTK  :", word_tokenize(text))

# 3. spaCy (production-grade classical)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("spaCy :", [token.text for token in doc])

# 4. Hugging Face (subword — what you'll actually build with)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("HF    :", tokenizer.tokenize(text))