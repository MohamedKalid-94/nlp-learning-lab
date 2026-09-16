import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Mohamed Kalid was born in 1994 On Sunday.")

for ent in doc.ents:
    print(f"{ent.text:15} -> {ent.label_}")