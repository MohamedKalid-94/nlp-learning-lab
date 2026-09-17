"""
Support Ticket Intelligence System
Module 1: Preprocessing — cleans raw ticket text before any downstream processing.
Reused from Phase-1-Classical-NLP/01-text-processing/main.py
"""

import re
import unicodedata


def preprocess(text: str) -> str:
    """
    Cleans raw ticket text: lowercase, strip URLs/emails/HTML,
    expand contractions, normalize accents, remove punctuation,
    collapse whitespace.
    """
    text = _remove_urls_emails(text)
    text = _remove_html_tags(text)
    text = text.lower()
    text = _expand_contractions(text)
    text = _normalize_unicode(text)
    text = _remove_punctuation(text)
    text = _collapse_whitespace(text)
    return text


def _remove_urls_emails(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    return text


def _remove_html_tags(text: str) -> str:
    return re.sub(r'<.*?>', '', text)


def _expand_contractions(text: str) -> str:
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "i'm": "i am", "it's": "it is", "isn't": "is not",
        "cant": "cannot", "im": "i am",  # ticket dataset has some typo'd forms too
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text


def _remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s#$.]', '', text)   # <- keeps # AND $ AND .

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


def _collapse_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


if __name__ == "__main__":
    # Quick test against a real ticket from the dataset
    sample = "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and it's been 2 weeks. I paid $89.99. Please refund me."
    print("BEFORE:", sample)
    print("AFTER: ", preprocess(sample))