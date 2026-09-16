from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Tiny training set
texts = [
    "win a free iphone now",
    "claim your free prize today",
    "meeting rescheduled to 3pm",
    "please review the attached report",
]
labels = ["spam", "spam", "not_spam", "not_spam"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X_train, labels)

# Test on new, unseen text
test_texts = ["free prize waiting for you", "let's schedule the report review"]
X_test = vectorizer.transform(test_texts)
predictions = model.predict(X_test)

for text, pred in zip(test_texts, predictions):
    print(f"{text:35} -> {pred}")