"""Intent discovery and baseline classification for AmazonHelp messages."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

SEED = 20260910
CUSTOMER_VALUE = "true"
MIN_CONFIDENCE = 0.45

INTENT_DESCRIPTIONS = {
    "delivery_tracking": "Where an order or package is, delivery status, or a tracking update.",
    "delivery_delay": "A late, missed, or promised delivery date problem.",
    "order_changes_cancellation": "Changing, cancelling, or correcting an order.",
    "returns_refunds": "Returning an item, requesting a refund, or asking about refund status.",
    "missing_wrong_damaged_item": "An item is missing, wrong, damaged, or not as described.",
    "account_login": "Sign-in, password, account access, or account security help.",
    "payment_billing": "Charges, payment methods, billing, gift cards, or payment failures.",
    "prime_subscription": "Prime membership, trial, renewal, or subscription benefits.",
    "other_or_unclear": "A customer message that does not match one clear support intent.",
}

INTENT_PATTERNS = {
    "account_login": (
        r"\b(password|sign in|signin|log in|login|logged in|account access|locked out|"
        r"verification code|two[- ]step|2[- ]step|hack(?:ed)?|security)\b",
    ),
    "returns_refunds": (
        r"\b(return|returns|returned|refund|refunded|money back|reimburse|reimbursement)\b",
    ),
    "missing_wrong_damaged_item": (
        r"\b(missing|wrong item|incorrect item|damaged|broken|defective|empty box|"
        r"not what i ordered|item not received)\b",
    ),
    "order_changes_cancellation": (
        r"\b(cancel|cancellation|cancelled|change my order|modify my order|wrong address|"
        r"change address|edit order)\b",
    ),
    "payment_billing": (
        r"\b(charged|charge|billing|payment|credit card|debit card|gift card|"
        r"promo code|promotional balance|declined|unauthorized charge)\b",
    ),
    "prime_subscription": (
        r"\b(prime|membership|subscription|free trial|renewal|renew|prime video)\b",
    ),
    "delivery_delay": (
        r"\b(late|delayed|delay|overdue|past the delivery|not arrived|still waiting|"
        r"where is my package|where is my order|delivery date missed)\b",
    ),
    "delivery_tracking": (
        r"\b(track|tracking|shipment|shipped|package|parcel|delivery|delivered|"
        r"arriving|order status)\b",
    ),
}


def load_customer_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [
            row
            for row in csv.DictReader(input_file)
            if row.get("inbound", "").strip().lower() == CUSTOMER_VALUE
        ]


def discover_intents(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Create transparent silver labels for discovery and baseline evaluation."""
    labeled_rows = []
    for row in rows:
        text = row.get("text", "").lower()
        matches = [
            intent for intent, patterns in INTENT_PATTERNS.items()
            if any(re.search(pattern, text) for pattern in patterns)
        ]
        label = matches[0] if len(matches) == 1 else "other_or_unclear"
        labeled_rows.append({**row, "intent": label, "label_source": "silver_keyword_rule"})
    return labeled_rows


def conversation_group(row: dict[str, str]) -> str:
    return (
        row.get("in_response_to_tweet_id", "").strip()
        or row.get("tweet_id", "").strip()
    )


def split_by_conversation_group(
    rows: list[dict[str, str]], seed: int, validation_fraction: float = 0.2
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted({conversation_group(row) for row in rows})
    random_generator = random.Random(seed)
    random_generator.shuffle(groups)
    validation_count = max(1, int(len(groups) * validation_fraction))
    validation_groups = set(groups[:validation_count])
    train_rows = [row for row in rows if conversation_group(row) not in validation_groups]
    validation_rows = [row for row in rows if conversation_group(row) in validation_groups]
    return train_rows, validation_rows


def train_classifier(rows: list[dict[str, str]]) -> Pipeline:
    classifier = Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=500, random_state=SEED
                ),
            ),
        ]
    )
    classifier.fit([row["text"] for row in rows], [row["intent"] for row in rows])
    return classifier


def evaluate_classifier(classifier: Pipeline, rows: list[dict[str, str]]) -> dict:
    labels = list(INTENT_DESCRIPTIONS)
    actual = [row["intent"] for row in rows]
    predicted = classifier.predict([row["text"] for row in rows])
    probabilities = classifier.predict_proba([row["text"] for row in rows])
    confidence = probabilities.max(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    pair_counts = Counter()
    for actual_label, predicted_label in zip(actual, predicted):
        if actual_label != predicted_label:
            pair_counts[(actual_label, predicted_label)] += 1
    return {
        "validation_rows": len(rows),
        "accuracy_against_silver_labels": round(sum(
            actual_label == predicted_label
            for actual_label, predicted_label in zip(actual, predicted)
        ) / len(rows), 4) if rows else 0.0,
        "macro_precision": round(float(precision.mean()), 4),
        "macro_recall": round(float(recall.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "per_intent": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
        "most_confused_pairs": [
            {"actual": actual_label, "predicted": predicted_label, "count": count}
            for (actual_label, predicted_label), count in pair_counts.most_common(10)
        ],
        "low_confidence_count": int((confidence < MIN_CONFIDENCE).sum()),
        "low_confidence_fraction": round(float((confidence < MIN_CONFIDENCE).mean()), 4),
    }


def representative_examples(rows: list[dict[str, str]], limit: int = 3) -> dict[str, list[dict[str, str]]]:
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if len(examples[row["intent"]]) < limit and row.get("text", "").strip():
            examples[row["intent"]].append(
                {
                    "tweet_id": row.get("tweet_id", ""),
                    "text": row["text"],
                }
            )
    return dict(examples)


def write_labeled_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_step2(input_path: Path, reports_dir: Path, labeled_output: Path, seed: int = SEED) -> dict:
    rows = load_customer_rows(input_path)
    if not rows:
        raise ValueError("No inbound customer rows found in the Step 2 input.")
    labeled_rows = discover_intents(rows)
    train_rows, validation_rows = split_by_conversation_group(labeled_rows, seed)
    classifier = train_classifier(train_rows)
    metrics = evaluate_classifier(classifier, validation_rows)
    label_counts = Counter(row["intent"] for row in labeled_rows)
    results = {
        "input": str(input_path),
        "seed": seed,
        "data_rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "taxonomy": INTENT_DESCRIPTIONS,
        "intent_frequencies": dict(label_counts),
        "representative_examples": representative_examples(train_rows),
        "metrics": metrics,
        "method": {
            "discovery": "Observed AmazonHelp inbound text was reviewed through recurring support vocabulary and transparent keyword rules.",
            "labeling": "Silver labels use ordered keyword rules; ambiguous multi-match rows become other_or_unclear.",
            "classifier": "TF-IDF word unigrams/bigrams with logistic regression.",
            "split": "80/20 deterministic conversation-group split; parent tweet IDs are never split across train and validation when available.",
            "leakage_protection": "Tweet IDs are excluded from features, and validation groups are selected before classifier fitting.",
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "step2_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "step2_intent_report.md").write_text(
        build_report(results), encoding="utf-8"
    )
    write_labeled_rows(labeled_rows, labeled_output)
    return results


def build_report(results: dict) -> str:
    metrics = results["metrics"]
    taxonomy = results["taxonomy"]
    frequencies = results["intent_frequencies"]
    examples = results["representative_examples"]
    taxonomy_lines = "\n".join(
        f"- **`{label}`** ({frequencies.get(label, 0):,} rows): {description}"
        for label, description in taxonomy.items()
    )
    example_lines = []
    for label, label_examples in examples.items():
        example_lines.append(f"### `{label}`")
        example_lines.extend(f"- `{example['tweet_id']}`: {example['text']}" for example in label_examples)
    confusion_lines = "\n".join(
        f"- `{item['actual']}` -> `{item['predicted']}`: {item['count']}"
        for item in metrics["most_confused_pairs"]
    ) or "- No disagreement pairs observed."
    return f"""# Step 2 Intent Discovery and Classification

## 1. Goal

Discover a useful first-pass taxonomy for AmazonHelp customer messages and train a reproducible baseline classifier. This is a development baseline, not a production accuracy claim.

## 2. Data Used

- Input: `{results['input']}`
- Customer rows: **{results['data_rows']:,}**
- Training rows: **{results['train_rows']:,}**
- Validation rows: **{results['validation_rows']:,}**
- Seed: **{results['seed']}**

## 3. How Intents Were Discovered

Recurring support vocabulary in the real AmazonHelp inbound sample was reviewed and grouped into operational support themes. Transparent keyword rules produced silver labels. Rows matching multiple themes were assigned `other_or_unclear`; no generated label is treated as human ground truth.

## 4. Final Taxonomy and 5. Intent Frequencies

{taxonomy_lines}

## 6. Intent Boundaries

The most difficult boundaries are delivery tracking versus delivery delay, returns/refunds versus missing/wrong/damaged item, and payment/billing versus order changes. Keyword rules are intentionally conservative and send ambiguous rows to `other_or_unclear`.

## 7. Context Strategy

The baseline classifies the customer message text only. Conversation metadata is used for grouping and leakage prevention, not as a predictive shortcut.

## 8. Classifier Approach

TF-IDF word unigrams and bigrams feed a logistic regression classifier. This is simple to inspect, deterministic, and appropriate as a first benchmark.

## 9. Train/Validation Split

An 80/20 deterministic split is performed by conversation group, using the parent tweet ID when available. This reduces the chance that near-duplicate turns from one conversation appear on both sides.

## 10. Leakage Prevention

Tweet IDs are excluded from features. Validation groups are selected before training. The labels are generated before the split and are explicitly reported as silver labels rather than ground truth.

## 11. Metrics

Metrics below measure agreement with held-out silver labels, not human annotation accuracy:

- Accuracy: **{metrics['accuracy_against_silver_labels']:.4f}**
- Macro precision: **{metrics['macro_precision']:.4f}**
- Macro recall: **{metrics['macro_recall']:.4f}**
- Macro F1: **{metrics['macro_f1']:.4f}**

## 12. Confusion Matrix

Label order: `{metrics['confusion_matrix_labels']}`.

```text
{json.dumps(metrics['confusion_matrix'])}
```

## 13. Error Analysis

Most disagreement pairs against silver labels:

{confusion_lines}

Representative training examples:

{chr(10).join(example_lines)}

## 14. Low-Confidence Cases

The classifier produced **{metrics['low_confidence_count']:,}** validation predictions below confidence **{MIN_CONFIDENCE}** ({metrics['low_confidence_fraction']:.2%}). These should be prioritized for manual review.

## 15. Limitations

- Silver keyword labels are not human ground truth and may miss paraphrases or multilingual cases.
- Held-out metrics therefore describe internal consistency, not production performance.
- Taxonomy coverage and intent boundaries need review by a human annotator.
- The current model uses message text only and does not yet use structured conversation context.

## 16. What Should Improve Later

Create a separately annotated evaluation set, revise labels from disagreement analysis, add representative multilingual examples, and compare this baseline with context-aware models. Do not use this report as evidence of production-level accuracy.
"""