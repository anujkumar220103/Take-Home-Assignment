"""Run Step 4 only when the golden set contains real human labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data.step2_intents import INTENT_DESCRIPTIONS
from .metrics import validate_human_labels


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a manually labelled Step 4 golden set.")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_set.csv"))
    arguments = parser.parse_args()
    try:
        rows = load_csv(arguments.golden)
        validate_human_labels(rows, set(INTENT_DESCRIPTIONS))
    except (FileNotFoundError, ValueError) as error:
        print(f"Evaluation is not ready: {error}")
        print("Complete the human label columns in data/golden/golden_set.csv, then rerun.")
        return 1
    print(f"{len(rows)} human-labelled rows are ready for evaluation.")
    print("Full evaluation orchestration is intentionally pending until labels are reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())