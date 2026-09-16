from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runner", "studies", "studying", "happily", "better"]

for word in words:
    print(f"{word:12} -> {stemmer.stem(word)}")