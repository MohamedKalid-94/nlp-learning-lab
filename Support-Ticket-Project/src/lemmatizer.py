"""
Support Ticket Intelligence System
Module 5: Lemmatization — normalizes ticket text before classification.
Reused from Phase-1-Classical-NLP/05-Lemmatization
Uses spaCy (not NLTK's WordNetLemmatizer) because spaCy auto-detects POS,
unlike NLTK which needs the POS tag passed in manually.
"""

import spacy

nlp = spacy.load("en_core_web_sm")


def lemmatize(text: str) -> str:
    """
    Lemmatizes text using spaCy — POS is auto-detected from context,
    so no manual pos= argument needed (unlike NLTK's WordNetLemmatizer
    from Phase-1 topic 5, which required specifying pos manually).
    """
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


if __name__ == "__main__":
    from preprocessing import preprocess

    test_tickets = [
        "I was charged twice for the same subscription and I am cancelling it.",
        "The app keeps crashing every time I open the camera feature.",
        "My orders were delayed and the products arrived damaged.",
    ]

    for ticket in test_tickets:
        cleaned = preprocess(ticket)
        lemmatized = lemmatize(cleaned)
        print("Cleaned:    ", cleaned)
        print("Lemmatized: ", lemmatized)
        print()