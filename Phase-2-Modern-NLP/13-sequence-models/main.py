import torch
import torch.nn as nn

# Tiny LSTM example: process a sequence of word embeddings
input_size = 10   # embedding dimension per word
hidden_size = 20  # size of the memory/hidden state
seq_len = 5       # 5 words in the sentence
batch_size = 1

lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# Fake input: 1 sentence, 5 words, each word a 10-dim vector
fake_sentence = torch.randn(batch_size, seq_len, input_size)

output, (hidden_state, cell_state) = lstm(fake_sentence)

print("Output shape:", output.shape)         # (1, 5, 20) - hidden state per word
print("Final hidden state:", hidden_state.shape)  # (1, 1, 20) - summary after reading whole sentence