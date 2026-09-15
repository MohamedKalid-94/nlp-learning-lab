import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("My Name is Walter Hartwell White, Damn It")

for token in doc:
    print(f"{token.text:8} -> {token.pos_}")