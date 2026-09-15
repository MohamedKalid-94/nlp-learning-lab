"""
01 - Text Preprocessing
Implementation of core text cleaning steps used before any NLP pipeline.
"""

import re
import unicodedata

# A deliberately messy sample text to prove each step works
sample_text = """
Hello!!! This is a SAMPLE text   with   extra   spaces.
Visit https://example.com or email me at test@example.com for info.
I don't think this is right... <b>Bold claim</b> — 100% guaranteed!!
Café résumé naïve  (accented characters test)
"""


def lowercase(text: str) -> str:
    return text.lower()


def remove_urls_emails(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)       # URLs
    text = re.sub(r'\S+@\S+\.\S+', '', text)            # emails
    return text


def remove_html_tags(text: str) -> str:
    return re.sub(r'<.*?>', '', text)


def expand_contractions(text: str) -> str:
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "i'm": "i am", "it's": "it is", "isn't": "is not",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


def normalize_unicode(text: str) -> str:
    # Converts accented chars (café -> cafe) via NFKD normalization
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')


def collapse_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def preprocess(text: str) -> str:
    """Full pipeline, applied in a sensible order."""
    text = remove_urls_emails(text)
    text = remove_html_tags(text)
    text = lowercase(text)
    text = expand_contractions(text)
    text = normalize_unicode(text)
    text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text


if __name__ == "__main__":
    print("=== BEFORE ===")
    print(repr(sample_text))
    print("\n=== AFTER ===")
    print(repr(preprocess(sample_text)))