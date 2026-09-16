from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

words = ["running", "unhappiness", "COVID-19", "tokenization", "asdfghjkl"]

for word in words:
    print(f"{word:15} -> {tokenizer.tokenize(word)}")