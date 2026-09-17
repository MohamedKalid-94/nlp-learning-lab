"""
Support Ticket Intelligence System — entry point.
Run: python main.py
"""

import sys
sys.path.insert(0, "src")

from pipeline import setup_classifier, process_ticket


def main():
    print("Training classifier on data/tickets.csv...")
    vectorizer, model = setup_classifier()
    print("Ready.\n")

    example_ticket = (
        "Hi, my order #48213 for the SmartWatch X2 hasn't arrived and "
        "it's been 2 weeks. I paid $89.99. Please refund me."
    )

    result = process_ticket(example_ticket, vectorizer, model)

    print("Input ticket:")
    print(f"  {example_ticket}\n")
    print("Structured output:")
    print(f"  Category:     {result['category']}")
    print(f"  Action verb:  {result['action_verb']}")
    print(f"  Entities:     {result['entities']}")
    print(f"  Cleaned text: {result['cleaned_text']}")


if __name__ == "__main__":
    main()