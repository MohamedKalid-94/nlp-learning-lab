from gensim.models import Word2Vec

sentences = [
    ["king", "queen", "royal", "throne", "crown"],
    ["man", "woman", "person", "human"],
    ["king", "man", "royal", "throne"],
    ["queen", "woman", "royal", "crown"],
    ["dog", "cat", "pet", "animal"],
    ["dog", "bark", "animal", "pet"],
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

print(model.wv.most_similar("king"))
print(model.wv["king"].shape)