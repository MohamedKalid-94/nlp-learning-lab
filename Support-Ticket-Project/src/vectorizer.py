"""
Support Ticket Intelligence System
Modules 8 & 9: TF-IDF and Bag of Words — numeric vectorization for classification.
Reused from Phase-1-Classical-NLP/08-TF-IDF and 09-Bag-of-Words

Both vectorizers are built for comparison — Module 10's classifier will
use TF-IDF by default, but BoW is kept available to demonstrate the
accuracy difference in the README.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def build_tfidf_vectorizer(texts: list[str]) -> tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    """Fits a TF-IDF vectorizer on the given texts, returns the fitted vectorizer + matrix."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_bow_vectorizer(texts: list[str]) -> tuple[CountVectorizer, "scipy.sparse.csr_matrix"]:
    """Fits a Bag-of-Words vectorizer on the given texts, returns the fitted vectorizer + matrix."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


if __name__ == "__main__":
    from preprocessing import preprocess
    from lemmatizer import lemmatize
    import pandas as pd

    # Load the real 100-ticket dataset
    df = pd.read_csv("data/tickets.csv")

    # Preprocess + lemmatize all tickets (matches what Module 10's
    # classifier will actually receive)
    processed_texts = [
        lemmatize(preprocess(text)) for text in df["ticket_text"]
    ]

    tfidf_vectorizer, tfidf_matrix = build_tfidf_vectorizer(processed_texts)
    bow_vectorizer, bow_matrix = build_bow_vectorizer(processed_texts)

    print("Dataset size:", len(processed_texts), "tickets")
    print("TF-IDF vocabulary size:", len(tfidf_vectorizer.vocabulary_))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("BoW vocabulary size:", len(bow_vectorizer.vocabulary_))
    print("BoW matrix shape:", bow_matrix.shape)

    # Show top TF-IDF terms for one sample ticket
    sample_idx = 0
    print(f"\nSample ticket: {df['ticket_text'][sample_idx]}")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    scores = tfidf_matrix[sample_idx].toarray()[0]
    top_terms = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 TF-IDF terms:")
    for term, score in top_terms:
        if score > 0:
            print(f"  {term}: {score:.3f}")