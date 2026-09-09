import csv
import tempfile
import unittest
from pathlib import Path

from src.data.step2_intents import (
    discover_intents,
    run_step2,
    split_by_conversation_group,
)


ROWS = [
    {
        "tweet_id": "1",
        "author_id": "customer-1",
        "inbound": "True",
        "text": "I need the tracking number for my shipment.",
        "in_response_to_tweet_id": "brand-1",
    },
    {
        "tweet_id": "2",
        "author_id": "customer-2",
        "inbound": "True",
        "text": "My order is late and has not arrived.",
        "in_response_to_tweet_id": "brand-2",
    },
    {
        "tweet_id": "3",
        "author_id": "customer-3",
        "inbound": "True",
        "text": "I need a refund for this return.",
        "in_response_to_tweet_id": "brand-3",
    },
    {
        "tweet_id": "4",
        "author_id": "customer-4",
        "inbound": "True",
        "text": "I cannot login because I forgot my password.",
        "in_response_to_tweet_id": "brand-4",
    },
]


class Step2Tests(unittest.TestCase):
    def test_discovery_assigns_expected_intents(self):
        labeled_rows = discover_intents(ROWS)
        self.assertEqual(
            [row["intent"] for row in labeled_rows],
            [
                "delivery_tracking",
                "delivery_delay",
                "returns_refunds",
                "account_login",
            ],
        )
        self.assertTrue(all(row["label_source"] == "silver_keyword_rule" for row in labeled_rows))

    def test_split_keeps_parent_groups_separate(self):
        rows = ROWS + [
            {
                **ROWS[0],
                "tweet_id": "5",
                "text": "Another question about tracking.",
            }
        ]
        train_rows, validation_rows = split_by_conversation_group(rows, seed=11)
        train_groups = {row["in_response_to_tweet_id"] for row in train_rows}
        validation_groups = {row["in_response_to_tweet_id"] for row in validation_rows}
        self.assertTrue(train_groups.isdisjoint(validation_groups))

    def test_pipeline_writes_labeled_rows_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sample.csv"
            fieldnames = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
            with input_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ROWS * 3)

            results = run_step2(
                input_path,
                root / "reports",
                root / "labeled.csv",
                seed=11,
            )

            self.assertEqual(results["data_rows"], 12)
            self.assertTrue((root / "reports" / "step2_intent_report.md").exists())
            with (root / "labeled.csv").open(encoding="utf-8", newline="") as labeled_file:
                labeled_rows = list(csv.DictReader(labeled_file))
            self.assertEqual(len(labeled_rows), 12)
            self.assertIn("intent", labeled_rows[0])


if __name__ == "__main__":
    unittest.main()