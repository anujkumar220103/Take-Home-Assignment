"""Human-agreement calculations."""

from __future__ import annotations


def cohen_kappa(first: list[str], second: list[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Agreement inputs must be non-empty and equal length.")
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    labels = set(first) | set(second)
    expected = sum(
        (first.count(label) / len(first)) * (second.count(label) / len(second))
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0