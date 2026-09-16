from sklearn.metrics import f1_score, precision_score, recall_score

# --- F1 Score example ---
y_true = ["spam", "not_spam", "spam", "spam", "not_spam"]
y_pred = ["spam", "not_spam", "not_spam", "spam", "not_spam"]

f1 = f1_score(y_true, y_pred, pos_label="spam")
precision = precision_score(y_true, y_pred, pos_label="spam")
recall = recall_score(y_true, y_pred, pos_label="spam")

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

# --- BLEU score example ---
from nltk.translate.bleu_score import sentence_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "sat", "on", "a", "mat"]

bleu = sentence_bleu(reference, candidate)
print(f"BLEU score: {bleu:.3f}")

# --- ROUGE score example ---
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(
    "the cat sat on the mat",
    "the cat is sitting on the mat"
)
print("ROUGE-1:", scores["rouge1"])
print("ROUGE-L:", scores["rougeL"])