"""Human-agreement calculations."""

from __future__ import annotations

import math


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


def rating_agreement(first: list[float], second: list[float]) -> dict:
    if len(first) != len(second) or not first:
        raise ValueError("Rating inputs must be non-empty and equal length.")
    differences = [abs(left - right) for left, right in zip(first, second)]
    exact = sum(difference == 0 for difference in differences)
    within_one_scale_step = sum(difference <= 0.25 for difference in differences)
    return {
        "joint_count": len(first),
        "mean_absolute_difference": sum(differences) / len(differences),
        "exact_agreement": exact / len(first),
        "within_one_scale_step": within_one_scale_step / len(first),
        "spearman_correlation": _spearman(first, second),
    }


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(set(first)) == 1 or len(set(second)) == 1:
        return None
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_mean = sum(first_ranks) / len(first_ranks)
    second_mean = sum(second_ranks) / len(second_ranks)
    numerator = sum((left - first_mean) * (right - second_mean) for left, right in zip(first_ranks, second_ranks))
    denominator_left = math.sqrt(sum((value - first_mean) ** 2 for value in first_ranks))
    denominator_right = math.sqrt(sum((value - second_mean) ** 2 for value in second_ranks))
    return numerator / (denominator_left * denominator_right) if denominator_left and denominator_right else None


def _average_ranks(values: list[float]) -> list[float]:
    sorted_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_values):
        end = index + 1
        while end < len(sorted_values) and sorted_values[end][1] == sorted_values[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[sorted_values[position][0]] = rank
        index = end
    return ranks