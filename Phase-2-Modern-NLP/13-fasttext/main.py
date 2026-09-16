from gensim.models import FastText

sentences = [
    ["king", "queen", "royal", "throne", "crown"],
    ["man", "woman", "person", "human"],
    ["king", "man", "royal", "throne"],
    ["queen", "woman", "royal", "crown"],
]

model = FastText(sentences, vector_size=50, window=3, min_count=1, sg=1)

# The real test: a word NEVER seen in training
print(model.wv.most_similar("kingdom"))  # "kingdom" wasn't in training data at all
print(model.wv["kingdom"].shape)