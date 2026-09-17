"""
Support Ticket Intelligence System
Module 6: POS Tagging — extracts the main action verb from a ticket.
Reused from Phase-1-Classical-NLP/06-POS-Tagging
The main action verb is a useful routing signal (e.g. "refund", "cancel", "delete")
that's more precise than the raw category alone.
"""

import spacy

nlp = spacy.load("en_core_web_sm")


def get_main_verb(text: str) -> str | None:
    """
    Finds the main ACTION verb in a ticket - prioritizes IMPERATIVE verbs
    (no explicit subject, e.g. "Please refund me") over other verbs, since
    imperatives are usually the actual customer request, especially in
    multi-sentence tickets where the request often isn't in sentence 1.
    """
    doc = nlp(text)
    wrapper_verbs = {"want", "need", "like", "would"}

    root_verbs = [token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]

    # First priority: an imperative verb (no nsubj/nsubjpass child) -
    # this is almost always the actual customer request
    for verb in root_verbs:
        has_subject = any(child.dep_ in ("nsubj", "nsubjpass") for child in verb.children)
        if not has_subject and verb.lemma_ not in wrapper_verbs:
            return verb.lemma_

    # Second priority: check each ROOT verb, unwrap "want/need to X" patterns
    for verb in root_verbs:
        if verb.lemma_ in wrapper_verbs:
            for child in verb.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    return child.lemma_
        else:
            return verb.lemma_

    # Fallback: first verb found anywhere
    for token in doc:
        if token.pos_ == "VERB":
            return token.lemma_

    return None


if __name__ == "__main__":
    from preprocessing import preprocess

    test_tickets = [
        "Please refund my order, the product arrived damaged.",
        "I want to cancel my subscription immediately.",
        "Can you delete my account and all my data?",
        "The app keeps crashing every time I open the camera.",
    ]

    for ticket in test_tickets:
        cleaned = preprocess(ticket)
        main_verb = get_main_verb(cleaned)
        print(f"{ticket}\n  -> main verb: {main_verb}\n")