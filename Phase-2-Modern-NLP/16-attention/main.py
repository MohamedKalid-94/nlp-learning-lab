import torch
import torch.nn.functional as F

# Simplified self-attention for a 4-word sentence, embedding dim = 8
torch.manual_seed(0)
seq_len, embed_dim = 4, 8
words = ["the", "cat", "sat", "mat"]

X = torch.randn(seq_len, embed_dim)  # fake word embeddings

# Random weight matrices for Query, Key, Value (normally learned during training)
Wq = torch.randn(embed_dim, embed_dim)
Wk = torch.randn(embed_dim, embed_dim)
Wv = torch.randn(embed_dim, embed_dim)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Attention scores: how much each word should attend to every other word
scores = Q @ K.T / (embed_dim ** 0.5)   # scaled dot-product
attention_weights = F.softmax(scores, dim=-1)

output = attention_weights @ V

print("Attention weights (each row sums to 1):")
print(attention_weights.round(decimals=2))
print("\nOutput shape:", output.shape)