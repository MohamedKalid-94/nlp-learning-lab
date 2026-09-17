"""
Support Ticket Intelligence System
Module 7: NER (Named Entity Recognition) — extracts structured entities.
Reused from Phase-1-Classical-NLP/07-NER, extended with regex fallback.

IMPORTANT: runs on the ORIGINAL (pre-preprocessing) ticket text, not the
cleaned/lowercased version — spaCy's entity recognizer relies on
capitalization and punctuation cues (e.g. "SmartWatch X2" as a proper
noun) that preprocessing would destroy.
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

# Regex patterns for structured entities that spaCy's general-purpose
# NER doesn't reliably catch (order numbers, money amounts) —
# this directly addresses the tokenizer fragmentation issue from Module 2/3
ORDER_PATTERN = re.compile(r'#\d+')
MONEY_PATTERN = re.compile(r'\$\d+\.?\d*')


def extract_entities(text: str) -> dict:
    """
    Extracts entities from RAW ticket text (not preprocessed).
    Combines spaCy's general NER (products, dates, misc proper nouns)
    with regex extraction for structured patterns spaCy tokenizers
    fragment (order numbers, money amounts).
    """
    doc = nlp(text)

    entities = {
        "orders": ORDER_PATTERN.findall(text),
        "money": MONEY_PATTERN.findall(text),
        "dates": [],
        "products": [],
        "other": [],
    }

    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ in ("PRODUCT", "ORG"):
            entities["products"].append(ent.text)
        elif ent.label_ not in ("CARDINAL", "MONEY"):  # avoid double-counting money via regex
            entities["other"].append((ent.text, ent.label_))

    return entities


if __name__ == "__main__":
    test_tickets = [
        "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and it's been 2 weeks. I paid $89.99. Please refund me.",
        "I was charged $45.00 twice for the same purchase on March 3rd. Order #77190. Need this fixed asap.",
    ]

    for ticket in test_tickets:
        print("Ticket:", ticket)
        entities = extract_entities(ticket)
        for key, value in entities.items():
            if value:
                print(f"  {key}: {value}")
        print()