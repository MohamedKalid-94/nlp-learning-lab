import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

words = [
    ("running", "v"),
    ("studies", "v"),
    ("better", "a"),   # a = adjective
    ("was", "v"),
]

for word, pos in words:
    print(f"{word:10} ({pos}) -> {lemmatizer.lemmatize(word, pos=pos)}")