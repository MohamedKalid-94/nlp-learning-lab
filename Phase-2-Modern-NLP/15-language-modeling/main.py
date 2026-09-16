import torch
import torch.nn.functional as F

# Simulate a model's predicted probability distribution over a tiny 5-word vocabulary
vocab = ["mat", "floor", "chair", "banana", "jupiter"]

# Model's predicted probabilities for "The cat sat on the ___"
predicted_probs = torch.tensor([0.5, 0.3, 0.15, 0.04, 0.01])

# The ACTUAL correct next word was "mat" (index 0)
actual_word_index = 0

# Cross-entropy loss for this one prediction
loss = -torch.log(predicted_probs[actual_word_index])
perplexity = torch.exp(loss)

print(f"Loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.4f}")

# Compare: what if the model was much less confident/wrong?
bad_predicted_probs = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.80])  # thinks "jupiter" is likely
bad_loss = -torch.log(bad_predicted_probs[actual_word_index])
bad_perplexity = torch.exp(bad_loss)

print(f"Bad model loss: {bad_loss.item():.4f}")
print(f"Bad model perplexity: {bad_perplexity.item():.4f}")