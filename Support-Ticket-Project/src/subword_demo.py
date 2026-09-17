"""
Support Ticket Intelligence System
Module 3: Subword Tokenization — DEMO ONLY, not used in the core pipeline.
Shows how a modern subword tokenizer would handle the same ticket text,
as a point of comparison against spaCy's word-level tokenizer (Module 2).
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def demo_subword_tokenize(text: str) -> list[str]:
    """
    Shows subword tokenization on the same cleaned ticket text,
    for comparison against Module 2's word-level tokens.
    Not used anywhere in pipeline.py — purely illustrative.
    """
    return tokenizer.tokenize(text)


if __name__ == "__main__":
    from preprocessing import preprocess

    sample = "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and it's been 2 weeks. I paid $89.99. Please refund me."
    cleaned = preprocess(sample)

    subword_tokens = demo_subword_tokenize(cleaned)

    print("Cleaned text:", cleaned)
    print("Subword tokens:", subword_tokens)
    print("\nCompare to Module 2's word tokens count (28) vs this count:", len(subword_tokens))