"""
Support Ticket Intelligence System
Module 4: Stemming — builds a stem-based keyword search index over tickets.
Reused from Phase-1-Classical-NLP/04-Stemming
"""

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def stem_tokens(tokens: list[str]) -> list[str]:
    """Stems a list of tokens — used to build a fast, approximate search index."""
    return [stemmer.stem(token) for token in tokens]


def build_stem_index(tickets: list[str]) -> dict[str, set[int]]:
    """
    Builds an inverted index: stem -> set of ticket indices containing that stem.
    Lets you quickly find "all tickets mentioning refund/refunding/refunds"
    by searching just the stem "refund" once.
    """
    from tokenizer import tokenize  # Module 2

    index: dict[str, set[int]] = {}
    for ticket_idx, ticket_text in enumerate(tickets):
        tokens = tokenize(ticket_text)
        stems = stem_tokens(tokens)
        for stem in stems:
            index.setdefault(stem, set()).add(ticket_idx)
    return index


def search_by_stem(query_word: str, index: dict[str, set[int]]) -> set[int]:
    """Given a search word, stems it and looks up matching ticket indices."""
    query_stem = stemmer.stem(query_word)
    return index.get(query_stem, set())


if __name__ == "__main__":
    from preprocessing import preprocess

    # Small test set — a few tickets containing "refund" in different forms
    test_tickets = [
        preprocess("Please refund my order, the product arrived damaged."),
        preprocess("I want a refund immediately, this is unacceptable."),
        preprocess("Can you tell me the refunding process for returns?"),
        preprocess("The app keeps crashing on my phone."),  # unrelated, should NOT match
    ]

    index = build_stem_index(test_tickets)

    # Search using a DIFFERENT word form than what's in the tickets
    results = search_by_stem("refunds", index)  # tickets don't literally contain "refunds"

    print("Search query: 'refunds'")
    print("Matching ticket indices:", results)
    print("\nMatching tickets:")
    for idx in results:
        print(f"  [{idx}] {test_tickets[idx]}")