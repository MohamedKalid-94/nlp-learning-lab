from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

text = "The cat sat on the"
inputs = tokenizer(text, return_tensors="pt")

# Generate 10 new tokens, one at a time (causal generation)
output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated:", generated_text)