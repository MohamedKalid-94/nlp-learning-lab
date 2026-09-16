from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "the cat sat on the mat",
    "the dog ran in the park",
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(docs)

feature_names = vectorizer.get_feature_names_out()

for doc_idx, doc in enumerate(docs):
    print(f"\nDoc {doc_idx + 1}: {doc}")
    counts = bow_matrix[doc_idx].toarray()[0]
    for word, count in zip(feature_names, counts):
        if count > 0:
            print(f"  {word:6} -> {count}")