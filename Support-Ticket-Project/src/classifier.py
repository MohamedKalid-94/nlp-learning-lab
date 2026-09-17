"""
Support Ticket Intelligence System
Module 10: Classification — trains a classifier on the real ticket dataset.
Reused from Phase-1-Classical-NLP/10-Text-Classification

Uses TF-IDF (not BoW) as the feature representation — TF-IDF generally
outperforms raw BoW counts for classification since it downweights
common words that appear across most categories.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


def train_classifier(texts: list[str], labels: list[str]):
    """
    Trains a TF-IDF + Naive Bayes classifier.
    Returns the fitted vectorizer and model, plus a held-out test report
    so you can see real accuracy, not just a toy demo.
    """
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    return vectorizer, model, report


def predict_category(text: str, vectorizer, model) -> str:
    """Predicts the category for a single new ticket."""
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]


if __name__ == "__main__":
    from preprocessing import preprocess
    from lemmatizer import lemmatize

    df = pd.read_csv("data/tickets.csv")
    processed_texts = [lemmatize(preprocess(text)) for text in df["ticket_text"]]
    labels = df["category"].tolist()

    vectorizer, model, report = train_classifier(processed_texts, labels)

    print("=== Classification Report (held-out test set) ===")
    print(report)

    # Test on a brand new, unseen ticket
    new_ticket = "I need my money back, the item I received was completely broken."
    processed_new = lemmatize(preprocess(new_ticket))
    prediction = predict_category(processed_new, vectorizer, model)

    print(f"New ticket: {new_ticket}")
    print(f"Predicted category: {prediction}")