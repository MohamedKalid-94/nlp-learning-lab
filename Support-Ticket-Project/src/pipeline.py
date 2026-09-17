"""
Support Ticket Intelligence System
pipeline.py — wires all 10 modules into one end-to-end function.
"""

import pandas as pd
from preprocessing import preprocess
from lemmatizer import lemmatize
from pos_tagger import get_main_verb
from entities import extract_entities
from classifier import train_classifier, predict_category


def process_ticket(raw_text: str, vectorizer, model) -> dict:
    """
    Runs a single raw ticket through the full pipeline:
    NER (raw text) -> preprocessing -> lemmatization -> POS (action verb)
    -> TF-IDF -> classification -> structured output
    """
    # NER needs the ORIGINAL text (capitalization/punctuation matter)
    entities = extract_entities(raw_text)

    # Everything else works on cleaned + lemmatized text
    cleaned = preprocess(raw_text)
    lemmatized = lemmatize(cleaned)

    main_verb = get_main_verb(cleaned)
    category = predict_category(lemmatized, vectorizer, model)

    return {
        "category": category,
        "action_verb": main_verb,
        "entities": entities,
        "cleaned_text": cleaned,
    }


def setup_classifier():
    """Loads the dataset and trains the classifier once (call this at startup)."""
    df = pd.read_csv("data/tickets.csv")
    processed_texts = [lemmatize(preprocess(text)) for text in df["ticket_text"]]
    labels = df["category"].tolist()
    vectorizer, model, _ = train_classifier(processed_texts, labels)
    return vectorizer, model


if __name__ == "__main__":
    vectorizer, model = setup_classifier()

    test_ticket = "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and it's been 2 weeks. I paid $89.99. Please refund me."

    result = process_ticket(test_ticket, vectorizer, model)

    print("Input:", test_ticket)
    print()
    print("=== Structured Output ===")
    print(f"Category:     {result['category']}")
    print(f"Action verb:  {result['action_verb']}")
    print(f"Entities:     {result['entities']}")
    print(f"Cleaned text: {result['cleaned_text']}")