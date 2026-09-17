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
    Finds the main ACTION verb in a ticket, not just the grammatical root.
    Skips common wrapper verbs (want/need/like) to find the real requested action.
    """
    doc = nlp(text)
    wrapper_verbs = {"want", "need", "like", "would"}

    root_verb = None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token
            break

    # If root verb is a "wrapper" verb, look for the real action verb
    # in its dependent clause (marked by dependency label "xcomp" - open clausal complement)
    if root_verb and root_verb.lemma_ in wrapper_verbs:
        for child in root_verb.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
                return child.lemma_

    if root_verb:
        return root_verb.lemma_

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