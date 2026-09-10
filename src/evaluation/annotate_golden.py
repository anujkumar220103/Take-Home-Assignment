"""Interactive, resumable manual annotation for the Step 4 golden set."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Callable, TextIO

from src.data.step2_intents import INTENT_DESCRIPTIONS

GOLDEN_PATH = Path("data/golden/golden_set.csv")
BACKUP_PATH = Path("data/golden/golden_set.backup.csv")
ESCALATION_OPTIONS = ("AUTO_HANDLE", "ESCALATE_TO_HUMAN")
QUALITY_SCORES = ("0", "0.25", "0.5", "0.75", "1")
REQUIRED_ANNOTATION_FIELDS = (
    "human_intent",
    "human_escalation",
    "human_reply_quality",
    "human_grounding_quality",
)


def load_golden_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("Golden CSV has no header row.")
        rows = list(reader)
    if not rows:
        raise ValueError("Golden CSV has no annotation rows.")
    missing_fields = [field for field in REQUIRED_ANNOTATION_FIELDS if field not in reader.fieldnames]
    if missing_fields:
        raise ValueError(f"Golden CSV is missing annotation fields: {missing_fields}")
    return list(reader.fieldnames), rows


def find_next_incomplete(rows: list[dict[str, str]]) -> int | None:
    for index, row in enumerate(rows):
        if row.get("annotation_status", "").strip().lower() != "complete":
            return index
    return None


def validate_annotation(annotation: dict[str, str]) -> None:
    if annotation.get("human_intent") not in INTENT_DESCRIPTIONS:
        raise ValueError("Choose one of the listed intent numbers.")
    if annotation.get("human_escalation") not in ESCALATION_OPTIONS:
        raise ValueError("Choose AUTO_HANDLE or ESCALATE_TO_HUMAN.")
    for field in ("human_reply_quality", "human_grounding_quality"):
        if annotation.get(field) not in QUALITY_SCORES:
            raise ValueError(f"{field} must be one of: {', '.join(QUALITY_SCORES)}")
    if annotation.get("annotation_status") != "complete":
        raise ValueError("Completed annotations must use annotation_status=complete.")


def render_example(row: dict[str, str], completed: int, total: int) -> str:
    """Render only human-review information; model and silver fields are excluded."""
    lines = [
        "Golden annotation",
        f"Completed: {completed} / {total}",
        f"Remaining: {total - completed}",
        "",
        f"Example: {row.get('example_id', '')}",
        "",
        "Customer message:",
        row.get("input_message", ""),
    ]
    historical_response = row.get("historical_response", "").strip()
    if historical_response:
        lines.extend(
            [
                "",
                "Available prior support response (for grounding review):",
                historical_response,
            ]
        )
    return "\n".join(lines)


def _prompt_choice(prompt: str, options: list[str], input_fn: Callable[[str], str]) -> str:
    while True:
        choice = input_fn(prompt).strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"Invalid selection. Enter a number from 1 to {len(options)}.")


def collect_annotation(input_fn: Callable[[str], str] = input) -> dict[str, str]:
    intent_options = list(INTENT_DESCRIPTIONS)
    print("\nHuman intent:")
    for number, intent in enumerate(intent_options, start=1):
        print(f"{number}. {intent} - {INTENT_DESCRIPTIONS[intent]}")
    human_intent = _prompt_choice("Intent number: ", intent_options, input_fn)

    print("\nShould this request be handled automatically?")
    print("1. AUTO_HANDLE")
    print("2. ESCALATE_TO_HUMAN")
    human_escalation = _prompt_choice(
        "Escalation number: ", list(ESCALATION_OPTIONS), input_fn
    )

    print("\nReply-quality score: 0 = unusable, 0.25 = mostly incorrect, 0.5 = mixed, 0.75 = useful, 1 = fully useful")
    human_reply_quality = _prompt_choice(
        "Reply-quality score: ", list(QUALITY_SCORES), input_fn
    )
    print("Grounding-quality score: 0 = unsupported, 0.25 = weak, 0.5 = mixed, 0.75 = mostly supported, 1 = fully supported")
    human_grounding_quality = _prompt_choice(
        "Grounding-quality score: ", list(QUALITY_SCORES), input_fn
    )
    annotator_notes = input_fn("Optional notes (press Enter to leave blank): ").strip()
    annotation = {
        "human_intent": human_intent,
        "human_escalation": human_escalation,
        "human_reply_quality": human_reply_quality,
        "human_grounding_quality": human_grounding_quality,
        "annotator_notes": annotator_notes,
        "annotation_status": "complete",
    }
    validate_annotation(annotation)
    return annotation


def save_annotation(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    row_index: int,
    annotation: dict[str, str],
    backup_path: Path,
) -> None:
    validate_annotation(annotation)
    if not backup_path.exists():
        shutil.copyfile(path, backup_path)
    updated_rows = [dict(row) for row in rows]
    updated_rows[row_index].update(annotation)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated_rows)
    rows[row_index].update(annotation)


def annotate_file(
    path: Path = GOLDEN_PATH,
    backup_path: Path = BACKUP_PATH,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> None:
    fieldnames, rows = load_golden_rows(path)
    while True:
        row_index = find_next_incomplete(rows)
        if row_index is None:
            print(f"{len(rows)} / {len(rows)} annotations complete.", file=output)
            return
        completed = len(rows) - sum(
            row.get("annotation_status", "").strip().lower() != "complete" for row in rows
        )
        print(render_example(rows[row_index], completed, len(rows)), file=output)
        annotation = collect_annotation(input_fn)
        save_annotation(path, fieldnames, rows, row_index, annotation, backup_path)
        completed += 1
        print(f"Saved {rows[row_index].get('example_id', '')}: {completed} / {len(rows)} complete.\n", file=output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually annotate one golden-set example at a time.")
    parser.add_argument("--input", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--backup", type=Path, default=BACKUP_PATH)
    arguments = parser.parse_args()
    try:
        annotate_file(arguments.input, arguments.backup)
    except EOFError:
        print("Annotation stopped before a label was saved.", file=sys.stderr)
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"Annotation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
