import torch
import torch.nn as nn

# Use PyTorch's built-in Transformer encoder layer (production-grade, not hand-rolled)
embed_dim = 8
num_heads = 2  # multi-head attention: 2 heads here
seq_len = 4

encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim, nhead=num_heads, dim_feedforward=16, batch_first=True
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # stack 2 layers

# Fake input: 1 sentence, 4 words, each 8-dim (already includes positional encoding conceptually)
fake_input = torch.randn(1, seq_len, embed_dim)

output = transformer_encoder(fake_input)

print("Input shape: ", fake_input.shape)
print("Output shape:", output.shape)