"""
Support Ticket Intelligence System
Module 2: Tokenization — splits cleaned ticket text into tokens.
Uses spaCy (production-grade, consistent with Modules 6/7 which also use spaCy)
"""

import spacy

# Load once at module level — loading the model is expensive,
# don't reload it every time tokenize() is called
nlp = spacy.load("en_core_web_sm")


def tokenize(text: str) -> list[str]:
    """
    Splits cleaned text into word tokens using spaCy.
    Using spaCy here (not regex/NLTK) because Modules 6 (POS) and 7 (NER)
    also need spaCy's Doc object — keeping one tokenizer consistent
    avoids subtle mismatches between modules.
    """
    doc = nlp(text)
    return [token.text for token in doc]


if __name__ == "__main__":
    from preprocessing import preprocess

    sample = "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and it's been 2 weeks. I paid $89.99. Please refund me."
    cleaned = preprocess(sample)
    tokens = tokenize(cleaned)

    print("Cleaned text:", cleaned)
    print("Tokens:", tokens)
    print("Token count:", len(tokens))