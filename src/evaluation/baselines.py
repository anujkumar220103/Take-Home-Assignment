"""Baselines evaluated only after human labels are complete."""

from __future__ import annotations

from collections import Counter

from src.data.step2_intents import discover_intents


def majority_baseline(train_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]]) -> list[str]:
    if not train_rows:
        raise ValueError("Training rows cannot be empty.")
    majority_intent = Counter(row["human_intent"] for row in train_rows).most_common(1)[0][0]
    return [majority_intent for _ in evaluation_rows]


def silver_keyword_baseline(evaluation_rows: list[dict[str, str]]) -> list[str]:
    """A transparent reference only; it never writes human labels."""
    return [row["intent"] for row in discover_intents(evaluation_rows)]