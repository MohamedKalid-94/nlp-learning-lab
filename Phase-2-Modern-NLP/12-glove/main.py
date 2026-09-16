import gensim.downloader as api

# Load small pretrained GloVe vectors (auto-downloads ~66MB)
glove_vectors = api.load("glove-wiki-gigaword-50")

print(glove_vectors.most_similar("king"))
print(glove_vectors["king"].shape)