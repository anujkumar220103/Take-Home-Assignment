"""Prepare an annotation-ready Step 4 golden-set file."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.agent.support_agent import build_cases, load_rows
from src.data.step2_intents import INTENT_DESCRIPTIONS, discover_intents

GOLDEN_FIELDS = (
    "example_id",
    "customer_tweet_id",
    "response_tweet_id",
    "input_message",
    "historical_response",
    "silver_sampling_group",
    "human_intent",
    "human_escalation",
    "human_reply_quality",
    "human_grounding_quality",
    "annotator_notes",
    "annotation_status",
)


def prepare_golden_set(
    raw_path: Path, output_path: Path, manifest_path: Path, size: int = 200, seed: int = 20260910
) -> dict:
    if size < 1:
        raise ValueError("Golden-set size must be positive.")
    cases = build_cases(raw_path)
    if len(cases) < size:
        raise ValueError(f"Only {len(cases)} usable cases are available; cannot sample {size}.")
    customer_rows = [row for row in load_rows(raw_path) if row.get("inbound", "").lower() == "true"]
    labels_by_text = {
        row["text"]: row["intent"]
        for row in discover_intents(customer_rows)
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for case in cases:
        # This is used only to spread annotation coverage, never as a gold label.
        grouped[labels_by_text.get(case["customer_text"], "other_or_unclear")].append(case)
    random_generator = random.Random(seed)
    for group in grouped.values():
        random_generator.shuffle(group)
    selected: list[dict[str, str]] = []
    groups = list(grouped.values())
    while len(selected) < size:
        added = False
        for group in groups:
            if group and len(selected) < size:
                selected.append(group.pop())
                added = True
        if not added:
            break
    selected.sort(key=lambda case: case["customer_tweet_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=GOLDEN_FIELDS)
        writer.writeheader()
        for index, case in enumerate(selected, start=1):
            writer.writerow(
                {
                    "example_id": f"golden-{index:04d}",
                    "customer_tweet_id": case["customer_tweet_id"],
                    "response_tweet_id": case["response_tweet_id"],
                    "input_message": case["customer_text"],
                    "historical_response": case["response_text"],
                    "silver_sampling_group": labels_by_text.get(
                        case["customer_text"], "other_or_unclear"
                    ),
                    "human_intent": "",
                    "human_escalation": "",
                    "human_reply_quality": "",
                    "human_grounding_quality": "",
                    "annotator_notes": "",
                    "annotation_status": "needs_human_label",
                }
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "golden_set_path": str(output_path),
                "size": len(selected),
                "seed": seed,
                "excluded_customer_tweet_ids": [case["customer_tweet_id"] for case in selected],
                "excluded_response_tweet_ids": [case["response_tweet_id"] for case in selected],
                "sampling_note": "silver_sampling_group is coverage metadata only, not a human label",
                "intent_options": list(INTENT_DESCRIPTIONS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"size": len(selected), "output": str(output_path), "manifest": str(manifest_path), "seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a blank human-annotation Step 4 golden set.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/twcs.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/golden/golden_set.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/golden/golden_manifest.json"))
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260910)
    arguments = parser.parse_args()
    try:
        result = prepare_golden_set(
            arguments.input, arguments.output, arguments.manifest, arguments.size, arguments.seed
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Golden-set preparation failed: {error}")
        return 1
    print(f"Prepared {result['size']} annotation rows at {result['output']}.")
    print("Human labels are still required before evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())