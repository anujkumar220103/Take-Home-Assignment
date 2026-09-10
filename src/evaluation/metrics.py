"""Small, auditable metrics for completed human annotations."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median


def validate_human_labels(rows: list[dict[str, str]], allowed_intents: set[str]) -> None:
    required = ("human_intent", "human_escalation", "human_reply_quality", "human_grounding_quality")
    allowed_escalations = {"AUTO_HANDLE", "ESCALATE_TO_HUMAN"}
    allowed_scores = {"0", "0.25", "0.5", "0.75", "1"}
    if not rows:
        raise ValueError("The golden set must contain at least one row.")
    for row in rows:
        missing = [field for field in required if not row.get(field, "").strip()]
        if missing:
            raise ValueError(f"Missing human labels for {row.get('example_id', 'unknown')}: {missing}")
        if row["human_intent"] not in allowed_intents:
            raise ValueError(f"Unknown human intent: {row['human_intent']}")
        if row["human_escalation"] not in allowed_escalations:
            raise ValueError(f"Unknown human escalation: {row['human_escalation']}")
        for field in ("human_reply_quality", "human_grounding_quality"):
            if row[field] not in allowed_scores:
                raise ValueError(f"Invalid {field}: {row[field]}")
        if row.get("annotation_status", "").strip().lower() != "complete":
            raise ValueError(f"Incomplete annotation: {row.get('example_id', 'unknown')}")
    example_ids = [row.get("example_id", "") for row in rows]
    if any(not example_id for example_id in example_ids) or len(set(example_ids)) != len(example_ids):
        raise ValueError("Golden-set example IDs must be present and unique.")


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


def escalation_metrics(actual: list[str], predicted: list[str]) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Escalation labels must be non-empty and equal length.")
    labels = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
    result = classification_metrics(actual, predicted, labels)
    false_auto_handle = sum(
        human == "ESCALATE_TO_HUMAN" and model == "AUTO_HANDLE"
        for human, model in zip(actual, predicted)
    )
    false_escalation = sum(
        human == "AUTO_HANDLE" and model == "ESCALATE_TO_HUMAN"
        for human, model in zip(actual, predicted)
    )
    human_escalations = actual.count("ESCALATE_TO_HUMAN")
    human_auto_handles = actual.count("AUTO_HANDLE")
    return {
        "decision_accuracy": result["accuracy"],
        "auto_handle_precision": result["per_label"]["AUTO_HANDLE"]["precision"],
        "auto_handle_recall": result["per_label"]["AUTO_HANDLE"]["recall"],
        "escalate_to_human_precision": result["per_label"]["ESCALATE_TO_HUMAN"]["precision"],
        "escalate_to_human_recall": result["per_label"]["ESCALATE_TO_HUMAN"]["recall"],
        "false_auto_handle_count": false_auto_handle,
        "false_auto_handle_rate": false_auto_handle / human_escalations if human_escalations else 0.0,
        "false_escalation_count": false_escalation,
        "false_escalation_rate": false_escalation / human_auto_handles if human_auto_handles else 0.0,
        "confusion_matrix": result["confusion_matrix"],
        "confusion_matrix_labels": labels,
    }


def quality_summary(rows: list[dict[str, str]], field: str, action_field: str | None = None) -> dict:
    values = [float(row[field]) for row in rows]
    summary = {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "distribution": dict(sorted(Counter(row[field] for row in rows).items(), key=lambda item: float(item[0]))),
    }
    if action_field:
        summary["by_model_action"] = {
            action: quality_summary(
                [row for row in rows if row.get(action_field) == action], field
            )
            for action in ("AUTO_HANDLE", "ESCALATE_TO_HUMAN")
        }
    return summary