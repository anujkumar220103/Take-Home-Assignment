"""Command-line entry point for the complete Step 1 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_step1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TWCS Step 1 data pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/twcs.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/recommended_brand.csv"),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/samples/development_sample.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        results = run_step1(
            input_path=arguments.input,
            reports_dir=arguments.reports_dir,
            processed_path=arguments.processed_output,
            sample_path=arguments.sample_output,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Step 1 failed: {error}")
        return 1

    print(f"Step 1 complete: {results['profiling']['row_count']:,} rows analyzed.")
    print(f"Recommended author ID: {results['recommendation']['recommended_author_id']}")
    print(f"Reports: {arguments.reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
