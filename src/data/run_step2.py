"""Command-line entry point for Step 2 intent discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from .step2_intents import run_step2


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and classify AmazonHelp intents.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples/development_sample.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=Path("data/processed/step2_customer_intents.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260910)
    arguments = parser.parse_args()
    try:
        results = run_step2(
            arguments.input,
            arguments.reports_dir,
            arguments.labeled_output,
            arguments.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Step 2 failed: {error}")
        return 1
    print(f"Step 2 complete: {results['data_rows']:,} customer rows labeled.")
    print(f"Reports: {arguments.reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())