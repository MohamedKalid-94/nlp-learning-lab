from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are pets",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

feature_names = vectorizer.get_feature_names_out()

for doc_idx, doc in enumerate(docs):
    print(f"\nDoc {doc_idx + 1}: {doc}")
    scores = tfidf_matrix[doc_idx].toarray()[0]
    for word, score in zip(feature_names, scores):
        if score > 0:
            print(f"  {word:8} -> {score:.3f}")