"""Baselines evaluated only after human labels are complete."""

from __future__ import annotations

from collections import Counter

from src.data.step2_intents import discover_intents, train_classifier


def majority_baseline(train_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]) -> list[str]:
    if not train_rows:
        raise ValueError("Training rows cannot be empty.")
    labels = [row.get("intent", row.get("human_intent", "")) for row in train_rows]
    if any(not label for label in labels):
        raise ValueError("Training rows must contain intent labels.")
    majority_intent = Counter(labels).most_common(1)[0][0]
    return [majority_intent for _ in evaluation_rows]


def simple_tfidf_baseline(train_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]) -> list[str]:
    """Train the plain Step 2 text classifier without Step 3 retrieval or reranking."""
    if not train_rows:
        raise ValueError("Training rows cannot be empty.")
    labeled_rows = discover_intents(train_rows)
    classifier = train_classifier(labeled_rows)
    return list(classifier.predict([row.get("input_message", row.get("text", "")) for row in evaluation_rows]))


def silver_keyword_baseline(evaluation_rows: list[dict[str, str]]) -> list[str]:
    """A transparent reference only; it never writes human labels."""
    return [row["intent"] for row in discover_intents(evaluation_rows)]