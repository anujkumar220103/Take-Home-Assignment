"""Small, auditable metrics for completed human annotations."""

from __future__ import annotations

from collections import Counter


def validate_human_labels(rows: list[dict[str, str]], allowed_intents: set[str]) -> None:
    required = ("human_intent", "human_escalation", "human_reply_quality", "human_grounding_quality")
    for row in rows:
        missing = [field for field in required if not row.get(field, "").strip()]
        if missing:
            raise ValueError(f"Missing human labels for {row.get('example_id', 'unknown')}: {missing}")
        if row["human_intent"] not in allowed_intents:
            raise ValueError(f"Unknown human intent: {row['human_intent']}")


def classification_metrics(actual: list[str], predicted: list[str], labels: list[str]) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted labels must be non-empty and equal length.")
    matrix = [[0 for _ in labels] for _ in labels]
    label_index = {label: index for index, label in enumerate(labels)}
    for actual_label, predicted_label in zip(actual, predicted):
        if actual_label not in label_index or predicted_label not in label_index:
            raise ValueError("Metrics received a label outside the declared label list.")
        matrix[label_index[actual_label]][label_index[predicted_label]] += 1
    accuracy = sum(actual_label == predicted_label for actual_label, predicted_label in zip(actual, predicted)) / len(actual)
    per_label = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        predicted_total = sum(row[index] for row in matrix)
        actual_total = sum(matrix[index])
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual_total}
    return {
        "accuracy": accuracy,
        "macro_precision": sum(item["precision"] for item in per_label.values()) / len(labels),
        "macro_recall": sum(item["recall"] for item in per_label.values()) / len(labels),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_label": per_label,
    }