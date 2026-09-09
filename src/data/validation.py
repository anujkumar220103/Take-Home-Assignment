"""Validation and lightweight schema checks for the TWCS input file."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REQUIRED_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)


@dataclass(frozen=True)
class ValidationResult:
    row_count: int
    columns: list[str]
    missing_required_columns: list[str]
    blank_value_counts: dict[str, int]

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_columns


def read_rows(input_path: Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows while keeping memory use bounded for large datasets."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("The input CSV has no header row.")
        for row in reader:
            yield {key: (value or "") for key, value in row.items() if key is not None}


def validate_input(input_path: Path) -> ValidationResult:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}. "
            "Place the raw file at data/raw/twcs.csv."
        )
    if input_path.stat().st_size == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        columns = reader.fieldnames or []
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in columns
        ]
        blank_value_counts = {column: 0 for column in columns}
        row_count = 0
        for row in reader:
            row_count += 1
            for column in columns:
                if not (row.get(column) or "").strip():
                    blank_value_counts[column] += 1

    if row_count == 0:
        raise ValueError(f"Input dataset has a header but no data rows: {input_path}")

    return ValidationResult(
        row_count=row_count,
        columns=columns,
        missing_required_columns=missing_columns,
        blank_value_counts=blank_value_counts,
    )
