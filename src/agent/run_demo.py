"""Run an observable Step 3 support-agent demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .support_agent import (
    answer_query,
    build_cases,
    build_retrieval_index,
    load_rows,
    write_index_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Step 3 AmazonHelp support agent demo.")
    parser.add_argument("query", nargs="?", default="Where is my package? It is late.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/twcs.csv"))
    parser.add_argument(
        "--classifier-input",
        type=Path,
        default=Path("data/samples/development_sample.csv"),
    )
    parser.add_argument("--index-summary", type=Path, default=Path("reports/step3_index.json"))
    parser.add_argument("--use-llm", action="store_true")
    arguments = parser.parse_args()
    try:
        classifier_rows = load_rows(arguments.classifier_input)
        customer_rows = [
            row for row in classifier_rows if row.get("inbound", "").lower() == "true"
        ]
        cases = build_cases(arguments.input)
        index = build_retrieval_index(cases, customer_rows)
        result = answer_query(arguments.query, index, customer_rows, arguments.use_llm)
        write_index_summary(arguments.index_summary, cases)
    except (FileNotFoundError, ValueError) as error:
        print(f"Step 3 failed: {error}")
        return 1
    output = json.dumps(
        {**result, "retrieved_cases": result["retrieved_cases"]},
        indent=2,
        ensure_ascii=True,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())