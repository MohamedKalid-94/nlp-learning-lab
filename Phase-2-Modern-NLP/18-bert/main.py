from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state = contextual embedding for every token
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("Output shape:", outputs.last_hidden_state.shape)