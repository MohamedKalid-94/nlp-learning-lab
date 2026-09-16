import torch
import math

def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

    pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
    return pe

seq_len, embed_dim = 4, 8
pe = positional_encoding(seq_len, embed_dim)

print("Positional encoding shape:", pe.shape)
print(pe.round(decimals=3))

# Demonstrate: same word embedding, different position -> different final vector
word_embedding = torch.randn(embed_dim)  # pretend this is "cat"'s embedding
cat_at_position_0 = word_embedding + pe[0]
cat_at_position_3 = word_embedding + pe[3]

print("\nSame word, different positions -> different final vectors:", 
      not torch.equal(cat_at_position_0, cat_at_position_3))